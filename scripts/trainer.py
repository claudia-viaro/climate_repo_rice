"""
Training script for the rice environment with action debugging.

"""

import os
import warnings

# Must be before any Ray import
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Hide warnings inside the main interpreter
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import logging

for name in ["ray", "ray.rllib", "ray.tune", "ray._private.deprecation"]:
    logging.getLogger(name).setLevel(logging.ERROR)
import gzip
import zipfile
import shutil
import json
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import psutil
import time
import numpy as np
import yaml

# Add repo root to sys.path
REPO_DIR = Path(__file__).resolve().parent.parent  # ~/RICE-N-exp
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from my_project.utils.fixed_paths import (
    REGION_YAMLS_DIR,
    OUTPUTS_DIR,
    OTHER_YAMLS_DIR,
    UTILS_DIR,
)

# from my_project.utils.desired_outputs import desired_outputs
# from my_project.utils.run_unittests import import_class_from_path
# from my_project.envs.opt_helper import save


# from rice import Rice
from my_project.configs.envs.rice_env import Rice

from my_project.configs.envs.scenarios import (
    OptimalMitigation,
    MinimalMitigation,
    BasicClub,
    ExportAction,
    CarbonLeakage,
    CarbonLeakageFixed,
    BasicClubTariffAmbition,
    MinimalMitigationActionWindow,
    OptimalMitigationActionWindow,
    BasicClubFixed,
    BasicClubAblateMasks,
    AsymmetricPartners,
    NewPartners,
    TradeAwarePartners,
    NegoReward,
)
import argparse
from collections import OrderedDict
from tqdm import tqdm
import datetime
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy.policy import Policy
import ray.rllib.utils.deprecation
import functools
import ray
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

ALGOS = {
    "A2C": A2CConfig,
    "PPO": PPOConfig,
    "SAC": SACConfig,
}

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from datetime import datetime
from ray.tune.logger import NoopLogger

print = functools.partial(print, flush=True)
logging.getLogger().setLevel(logging.WARNING)

# scenarios
SCENARIO_MAPPING = {
    "default": Rice,
    "OptimalMitigation": OptimalMitigation,
    "MinimalMitigation": MinimalMitigation,
    "BasicClub": BasicClub,
    "ExportAction": ExportAction,
    "CarbonLeakage": CarbonLeakage,
    "CarbonLeakageFixed": CarbonLeakageFixed,
    "BasicClubTariffAmbition": BasicClubTariffAmbition,
    "MinimalMitigationActionWindow": MinimalMitigationActionWindow,
    "OptimalMitigationActionWindow": OptimalMitigationActionWindow,
    "BasicClubFixed": BasicClubFixed,
    "BasicClubAblateMasks": BasicClubAblateMasks,
    "AsymmetricPartners": AsymmetricPartners,
    "NewPartners": NewPartners,
    "TradeAwarePartners": TradeAwarePartners,
    "NegoReward": NegoReward,
}


class Callbacks(DefaultCallbacks):
    def __init__(
        self,
        current_iteration=1,
        save_iteration=None,
        save_enabled=True,
        save_dir=None,
        file_name=None,
        max_iterations=None,
    ):
        super().__init__()
        self.save_iteration = save_iteration  # RLlib iteration to trigger saving
        self.save_enabled = save_enabled
        self.save_dir = save_dir
        self.file_name = file_name
        self.max_iterations = max_iterations
        self.episode_counter = 0
        self.current_iteration = current_iteration  # RLlib iteration number
        self.reward_hist = {}
        self.reward_hist_full = {}

    def set_current_iteration(self, iteration: int):
        self.current_iteration = iteration
        print(
            f"[DEBUG] Callback iteration updated to {self.current_iteration} from main loop"
        )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        self.episode_counter += 1

        # Only save if enabled and we are at the requested RLlib iteration
        if self.save_enabled and (
            self.save_iteration is None or self.current_iteration == self.save_iteration
        ):
            env = worker.env.env  # unwrap to underlying env

            print(
                f"[DEBUG] Saving triggered at iteration {self.current_iteration}, episode {self.episode_counter}"
            )

            # Check RL info
            if hasattr(env, "last_info_rl") and env.last_info_rl:

                save_jsonl(
                    save_dir=self.save_dir,
                    file_name=f"{self.file_name}_rl_episode_iter{self.current_iteration}",
                    entries=env.last_info_rl,
                    suffix="jsonl",
                    append=True,
                )
            else:
                print("[DEBUG] No RL info to save for this episode")

            # Check Diffs info
            if hasattr(env, "last_info_diffs") and env.last_info_diffs:

                save_jsonl(
                    save_dir=self.save_dir,
                    file_name=f"{self.file_name}_diffs_episode_iter{self.current_iteration}",
                    entries=env.last_info_diffs,
                    suffix="jsonl",
                    append=True,
                )
            else:
                print("[DEBUG] No diff info to save for this episode")

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Called after each training iteration.
        Aggregates stats across policies, prints them, and saves them.
        Works for A2C/PPO/SAC.
        """
        env = algorithm.workers.local_worker().env.env

        per_policy_stats = []

        # Loop over all policies
        for policy_id, stats in result["info"].get("learner", {}).items():
            ls = stats.get("learner_stats", {})

            # Save per-policy stats
            policy_stats = {
                "iteration": result["training_iteration"],
                "policy_id": policy_id,
            }
            policy_stats.update(ls)

            # Optionally add environment info
            if hasattr(env, "last_info_diffs") and env.last_info_diffs:
                last_climate_step = env.last_info_diffs[-1]
                policy_stats["temp_rise_rl"] = last_climate_step.get(
                    "temp_rise_rl", None
                )

            per_policy_stats.append(policy_stats)

        # Aggregate stats across policies
        aggregated_stats = {"iteration": result["training_iteration"]}
        if per_policy_stats:
            all_keys = set(
                k
                for d in per_policy_stats
                for k in d.keys()
                if k not in ["iteration", "policy_id"]
            )
            for k in all_keys:
                vals = [d.get(k) for d in per_policy_stats if d.get(k) is not None]
                aggregated_stats[k] = float(np.mean(vals)) if vals else None

        # Print aggregate stats to console
        print(
            f"[Iteration {result['training_iteration']}] Aggregate learner stats: {aggregated_stats}"
        )

        # Save per-policy stats to JSONL
        if self.save_enabled:
            save_jsonl(
                save_dir=self.save_dir,
                file_name=f"{self.file_name}_learner_stats_per_policy",
                entries=per_policy_stats,
                suffix="jsonl",
                append=True,
            )

        # Save aggregated stats to JSONL
        if self.save_enabled:
            save_jsonl(
                save_dir=self.save_dir,
                file_name=f"{self.file_name}_learner_stats_aggregated",
                entries=[aggregated_stats],
                suffix="jsonl",
                append=True,
            )

        # Track per-agent rewards (economic utility)
        last_episode_rewards = (
            env.get_last_episode_rewards()
        )  # return dict {agent_id: reward}

        # Moving averages for per-agent rewards
        if not hasattr(self, "reward_hist"):
            self.reward_hist = {aid: [] for aid in last_episode_rewards.keys()}
        for aid, r in last_episode_rewards.items():
            self.reward_hist.setdefault(aid, []).append(r)
        # Track full reward history for plotting
        if not hasattr(self, "reward_hist_full"):
            self.reward_hist_full = {aid: [] for aid in last_episode_rewards.keys()}

        for aid, r in last_episode_rewards.items():
            self.reward_hist_full.setdefault(aid, []).append(r)
        # Aggregate reward across agents
        # Aggregate reward across agents
        agent_rewards = np.array(list(last_episode_rewards.values()))
        avg_reward = agent_rewards.mean()
        std_reward = agent_rewards.std()
        print(
            f"Iteration {result['training_iteration']}: avg_reward={avg_reward:.3f} ± {std_reward:.3f}"
        )

        # Aggregate learner stats across policies (losses)
        agg_policy_loss = []
        agg_vf_loss = []
        for pid, stats in result["info"]["learner"].items():
            ls = stats["learner_stats"]
            if ls.get("policy_loss") is not None:
                agg_policy_loss.append(ls["policy_loss"])
            if ls.get("vf_loss") is not None:
                agg_vf_loss.append(ls["vf_loss"])
        avg_policy_loss = np.mean(agg_policy_loss) if agg_policy_loss else None
        avg_vf_loss = np.mean(agg_vf_loss) if agg_vf_loss else None

        # Print summary
        print(
            f"Iteration {result['training_iteration']}: "
            f"avg_reward={avg_reward:.3f}, "
            f"policy_loss={avg_policy_loss:.3f}, vf_loss={avg_vf_loss:.3f}"
        )

        # Print per-agent reward
        for aid, hist in self.reward_hist.items():
            print(f"  Agent {aid}: mean_reward_last5iters={np.mean(hist):.3f}")
        # --- DEBUG PRINT ---
        print("[DEBUG] reward_hist_full after iteration", result["training_iteration"])
        for aid, hist in self.reward_hist_full.items():
            print(f"  Agent {aid}: {hist}")


def redirect_worker_logs(worker):

    try:
        log_path = f"/tmp/worker_{worker.worker_index}.log"
        f = open(log_path, "w", buffering=1)  # line-buffered
        sys.stdout = f
        sys.stderr = f
        print(f"[Worker {worker.worker_index}] Redirected logs to {log_path}")
    except Exception as e:
        print(
            f"[Worker {worker.worker_index} ERROR] Failed to redirect logs: {e}",
            file=sys.__stdout__,
        )


from ray.rllib.algorithms.callbacks import DefaultCallbacks
import copy


def plot_reward_convergence(reward_hist: dict, save_dir: str):
    """
    Plot per-agent rewards and aggregate mean ± 95% CI over iterations.
    Handles np.nan for missing iterations.
    """
    # Check if there is any data at all
    if all(len(v) == 0 for v in reward_hist.values()):
        print("[WARN] Reward history is empty. No data to plot.")
        return

    iterations = np.arange(1, max(len(v) for v in reward_hist.values()) + 1)
    plt.figure(figsize=(8, 5))

    # Per-agent reward lines
    for aid, hist in reward_hist.items():
        y = np.array(hist, dtype=float)
        if len(y) < len(iterations):
            y = np.pad(y, (0, len(iterations) - len(y)), constant_values=np.nan)
        plt.plot(iterations, y, label=f"Agent {aid}")

    # Aggregate mean ± 95% CI
    rewards_array = np.array(
        [
            np.pad(
                np.array(h, dtype=float),
                (0, len(iterations) - len(h)),
                constant_values=np.nan,
            )
            for h in reward_hist.values()
        ]
    )
    mean_reward = np.nanmean(rewards_array, axis=0)
    num_agents = rewards_array.shape[0]
    std_err = np.nanstd(rewards_array, axis=0) / np.sqrt(num_agents)
    ci95 = std_err * stats.t.ppf(0.975, df=num_agents - 1)

    plt.fill_between(
        iterations,
        mean_reward - ci95,
        mean_reward + ci95,
        alpha=0.2,
        color="k",
        label="95% CI",
    )
    plt.plot(iterations, mean_reward, "k--", label="Mean reward")

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Reward convergence per agent")
    plt.legend()
    plt.xticks(iterations)  # integer x-axis
    plt.tight_layout()

    save_path = os.path.join(save_dir, "reward_convergence.png")
    plt.savefig(save_path)
    print(f"[INFO] Reward convergence plot saved to {save_path}")


def get_config_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise ValueError(f"The run configuration is missing. Please check: {yaml_path}")
    with open(yaml_path, "r", encoding="utf8") as fp:
        return yaml.safe_load(fp)


_BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        spaces.Dict: A dictionary of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for key, val in obs.items():
        # list of lists are 'listified' np arrays
        _val = val
        if isinstance(val, list):
            _val = np.array(val)
        elif isinstance(val, (int, np.integer, float, np.floating)):
            _val = np.array([val])

        # assign Space
        if isinstance(_val, np.ndarray):
            large_num = float(_BIG_NUMBER)
            box = Box(
                low=-large_num,
                high=large_num,
                shape=_val.shape,
                dtype=np.float32,
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num,
                    high=large_num,
                    shape=_val.shape,
                    dtype=np.float32,
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[key] = box

        elif isinstance(_val, dict):
            dict_of_spaces[key] = recursive_obs_dict_to_spaces_dict(_val)
        else:
            raise TypeError
    return Dict(dict_of_spaces)


def recursive_list_to_np_array(dictionary):
    """
    Numpy-ify dictionary object to be used with RLlib.
    """
    if isinstance(dictionary, dict):
        new_d = {}
        for key, val in dictionary.items():
            if isinstance(val, list):
                new_d[key] = np.array(val)
            elif isinstance(val, dict):
                new_d[key] = recursive_list_to_np_array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                new_d[key] = np.array([val])
            elif isinstance(val, np.ndarray):
                new_d[key] = val
            else:
                raise AssertionError
        return new_d
    raise AssertionError


class EnvWrapper(MultiAgentEnv):
    """
    Environment wrapper for Rice, collects debug prints without modifying Rice.
    """

    def __init__(self, env_config=None):
        super().__init__()

        env_config = env_config or {}
        env_config_copy = env_config.copy()

        scenario = env_config_copy.pop("scenario", None)
        assert scenario in SCENARIO_MAPPING, f"Unknown scenario: {scenario}"
        env_config_copy.pop("logs_dir", None)
        env_config_copy.pop("file_name", None)
        env_config_copy.pop("save_dir", None)
        env_config_copy.pop("regions", None)
        # Instantiate the Rice environment only with expected args
        self.env = SCENARIO_MAPPING[scenario](**env_config_copy)

        self.action_space = self.env.action_space

        # Initialize observation space
        obs, info = self.env.reset()
        self.observation_space = recursive_obs_dict_to_spaces_dict(obs)

        # Agent bookkeeping
        self.agent_ids = list(range(self.env.num_regions))
        self.num_agents = len(self.agent_ids)

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset()
        super().reset(seed=seed)
        return recursive_list_to_np_array(obs), info

    def step(self, actions=None):
        """Step through the environment and optionally print debug info."""
        assert actions is not None
        assert isinstance(actions, dict)

        obs, rew, terminateds, truncateds, info = self.env.step(actions)

        return (
            recursive_list_to_np_array(obs),
            rew,
            terminateds,
            truncateds,
            info,
        )


def get_rllib_config(config_yaml=None, env_class=None, seed=None, save_dir=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert config_yaml is not None
    assert env_class is not None
    env_config = dict(config_yaml["env"])
    env_config["region_yamls_path"] = str(REGION_YAMLS_DIR)

    # Attach logs_dir if save_dir was passed
    if save_dir is not None:
        logs_dir = Path(save_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        env_config["logs_dir"] = str(logs_dir)

    env_object = create_env_object(env_class, env_config)
    multiagent_policies_config = get_multiagent_policies_config(
        config_yaml=config_yaml, env_object=env_object
    )

    trainer_config = get_trainer_config(config_yaml)

    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": env_config,  # config_yaml["env"],
        "framework": trainer_config["framework"],
        "multiagent": multiagent_policies_config,
        "num_workers": trainer_config["num_workers"],
        "num_gpus": trainer_config["num_gpus"],
        "num_cpus_per_worker": trainer_config["num_cpus_per_worker"],
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
        "train_batch_size": trainer_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config


def get_env_config(config_yaml):
    env_config = config_yaml["env"].copy()
    return env_config


def create_env_object(env_class, env_config):
    env_object = env_class(env_config=env_config)
    return env_object


def get_multiagent_policies_config(config_yaml=None, env_object=None):
    """
    Create a separate policy for each agent in the environment.
    """
    assert config_yaml is not None
    assert env_object is not None

    regions_policy_config = config_yaml["policy"]["regions"]

    num_agents = config_yaml["env"]["regions"]["num_agents"]
    policies = {}

    # Create a unique policy for each agent
    for agent_id in range(num_agents):
        policy_name = f"agent_{agent_id}"
        policies[policy_name] = (
            None,  # None = use default Torch policy (your TorchLinear)
            env_object.observation_space[agent_id],
            env_object.action_space[str(agent_id)],
            regions_policy_config,
        )

    # Map each agent to its own policy
    def policy_mapping_fn(agent_id=None, *args, **kwargs):
        assert agent_id is not None
        return f"agent_{agent_id}"

    multiagent_config = {
        "policies": policies,
        "policies_to_train": None,  # train all agents
        "policy_mapping_fn": policy_mapping_fn,
    }

    return multiagent_config


def get_trainer_config(config_yaml):
    trainer_config = config_yaml["trainer"]
    return trainer_config


def save_model_checkpoint(trainer_obj=None, save_directory=None, current_timestep=0):
    """
    Save trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to save the checkpoints."
    )
    model_params = trainer_obj.get_weights()
    for policy in model_params:
        filepath = os.path.join(
            save_directory,
            f"{policy}_{current_timestep}.state_dict",
        )

        torch.save(model_params[policy], filepath)


def zip_run_outputs(save_dir, file_name):
    """
    Zip the run outputs (checkpoints + logs) into a single archive.
    """
    if not os.path.exists(save_dir):
        print(f"[WARN] save_dir does not exist: {save_dir}")
        return

    zip_path = os.path.join(save_dir, f"{file_name}.zip")
    print(f"[DEBUG] Creating zip archive at: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Preserve folder structure relative to save_dir
                rel_path = os.path.relpath(file_path, save_dir)
                zf.write(file_path, rel_path)
                print(f"[DEBUG] Adding {file_path} as {rel_path} to zip")

    print(f"✅ Run outputs zipped to: {zip_path}")


def load_model_checkpoints(trainer_obj=None, save_directory=None, ckpt_idx=-1):
    """
    Load trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to load the checkpoints from."
    )
    files = [f for f in os.listdir(save_directory) if f.endswith("state_dict")]

    assert len(files) == len(trainer_obj.config["multiagent"]["policies"])

    model_params = trainer_obj.get_weights()
    for policy in model_params:
        policy_models = [
            os.path.join(save_directory, file) for file in files if policy in file
        ]
        # If there are multiple files, then use the ckpt_idx to specify the checkpoint
        assert ckpt_idx < len(policy_models)
        sorted_policy_models = sorted(policy_models, key=os.path.getmtime)
        policy_model_file = sorted_policy_models[ckpt_idx]
        model_params[policy] = torch.load(policy_model_file)
        # print(f"Loaded model checkpoints {policy_model_file}.")

    trainer_obj.set_weights(model_params)


def apply_algo_specific_params(config, policy_config, algo_name):
    """
    Apply algorithm-specific hyperparameters to the RLlib config object.

    Args:
        config: RLlib Config object (A2CConfig, PPOConfig, SACConfig)
        policy_yaml: YAML dict under `policy.regions`
        algo_name: str, algorithm name
    """
    # Common hyperparameters
    if "gamma" in policy_config:
        config = config.training(gamma=policy_config["gamma"])
    if "lr" in policy_config:
        config = config.training(lr=policy_config["lr"])

    # Gradient clipping
    if "max_grad_norm" in policy_config:
        # RLlib uses `grad_clip` in newer configs
        config = config.training(grad_clip=policy_config["max_grad_norm"])

    # Algorithm-specific settings
    if algo_name == "A2C":
        if "vf_loss_coeff" in policy_config:
            config = config.training(vf_loss_coeff=policy_config["vf_loss_coeff"])
        if "entropy_coeff_schedule" in policy_config:
            config = config.training(
                entropy_coeff_schedule=policy_config["entropy_coeff_schedule"]
            )

    elif algo_name == "PPO":
        if "vf_loss_coeff" in policy_config:
            config = config.training(vf_loss_coeff=policy_config["vf_loss_coeff"])
        if "entropy_coeff_schedule" in policy_config:
            config = config.training(
                entropy_coeff_schedule=policy_config["entropy_coeff_schedule"]
            )
        if "clip_param" in policy_config:
            config = config.training(clip_param=policy_config["clip_param"])
        if "lambda_" in policy_config:
            config = config.training(lambda_=policy_config["lambda_"])

    elif algo_name == "SAC":
        if "tau" in policy_config:
            config = config.training(tau=policy_config["tau"])
        if "target_entropy" in policy_config:
            config = config.training(target_entropy=policy_config["target_entropy"])
        if "alpha" in policy_config:
            config = config.training(alpha=policy_config["alpha"])

    return config


def create_trainer(
    config_yaml=None, save_dir=None, file_name=None, num_iters=None, seed=None
):
    """
    Create the RLlib trainer.
    """

    # Create the A2C trainer.
    # config_yaml["env"]["source_dir"] = source_dir

    from my_project.training.torch_models_discrete import TorchLinear

    rllib_config = get_rllib_config(
        config_yaml=config_yaml,
        env_class=EnvWrapper,
        seed=seed,
    )

    # Algorithm-specific parameters from YAML
    algo_name = config_yaml["trainer"].get("algorithm", "A2C")
    config_class = ALGOS[algo_name]
    config = config_class()

    # --- Multi-agent config ---
    config = config.multi_agent(
        policies=rllib_config["multiagent"]["policies"],
        policy_mapping_fn=rllib_config["multiagent"]["policy_mapping_fn"],
        policies_to_train=rllib_config["multiagent"]["policies_to_train"],
    )

    # --- Training, rollout, resources ---
    config = config.training(train_batch_size=rllib_config["train_batch_size"])
    config = config.resources(num_gpus=rllib_config["num_gpus"])
    config = config.rollouts(
        num_rollout_workers=rllib_config["num_workers"],
        num_envs_per_worker=rllib_config["num_envs_per_worker"],
        rollout_fragment_length=rllib_config["env_config"].get(
            "episode_length", 100
        ),  # set rollout_fragment_length = episode_length
    )
    config = config.framework(rllib_config["framework"])
    config = config.environment(
        EnvWrapper,
        env_config=rllib_config["env_config"],
        disable_env_checking=True,
    )

    # --- Callbacks for logging/saving ---
    save_cfg = config_yaml.get("saving", {})
    config = config.callbacks(
        lambda: Callbacks(
            save_iteration=save_cfg.get("save_iteration", None),
            save_enabled=save_cfg.get("save_climate_info", True),
            save_dir=save_dir,
            file_name=file_name,
            max_iterations=config_yaml["trainer"].get("num_episodes", None),
        )
    )

    config.seed = seed

    # --- Apply algorithm-specific hyperparameters from YAML ---
    policy_config = config_yaml.get("policy", {})
    config = apply_algo_specific_params(config, policy_config, algo_name)

    # --- Build and return RLlib trainer ---
    rllib_trainer = config.build()
    return rllib_trainer


def create_save_dir_path(results_dir=None):
    """
    Creates the save directory inside OUTPUTS_DIR with subfolders
    for checkpoints and logs.
    """
    if results_dir is None:
        # Use timestamp as the default folder name
        results_dir = f"{int(time.time())}"

    results_save_dir = OUTPUTS_DIR / results_dir
    results_save_dir.mkdir(parents=True, exist_ok=True)

    (results_save_dir / "checkpoints").mkdir(exist_ok=True)
    (results_save_dir / "logs").mkdir(exist_ok=True)

    return results_save_dir


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(
            obj, (np.void)
        ):  # Catch-all for any other types not explicitly handled
            return None
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def fetch_episode_states(trainer_obj=None, episode_states=None, file_name=None):
    """
    Roll out the environment for one episode and record state only at the end of each full
    3-stage cycle (i.e., after climate/economy step).
    """
    assert trainer_obj is not None
    assert episode_states is not None and isinstance(episode_states, list)
    assert len(episode_states) > 0

    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    obs, _ = env_object.reset()
    env = env_object.env

    # Setup outputs: record only ~100 economic/climate steps (plus 1 final state)
    xN = env.common_params["xN"]
    outputs = {
        state: np.nan * np.ones((xN + 1,) + env.global_state[state]["value"].shape[1:])
        for state in episode_states
    }

    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_agents):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    logical_timestep = 0  # Only incremented at each climate/economy step

    for env_step in range(env.episode_length):
        # Compute actions for all agents
        actions = {}
        for region_id in range(env.num_agents):
            if len(agent_states[region_id]) == 0:
                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    state=agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
            else:
                actions[region_id], agent_states[region_id], _ = (
                    trainer_obj.compute_actions(
                        obs[region_id],
                        state=agent_states[region_id],
                        policy_id=policy_ids[region_id],
                    )
                )

        # Step the environment
        obs, rewards, done, truncateds, info = env_object.step(actions)

        # Record only after full 3-stage cycle (when climate/economy step has occurred)
        if env.negotiation_stage == 0:
            for state in episode_states:
                outputs[state][logical_timestep] = env.global_state[state]["value"][
                    env.current_timestep
                ]
            logical_timestep += 1

        # Final state at termination
        if done["__all__"]:
            for state in episode_states:
                outputs[state][logical_timestep] = env.global_state[state]["value"][
                    env.current_timestep
                ]
            break

    # Optionally save to JSON
    if file_name:
        from datetime import datetime
        import os, json

        current_directory = os.path.dirname(__file__)
        eval_directory = os.path.abspath(os.path.join(current_directory, "..", "evals"))
        os.makedirs(eval_directory, exist_ok=True)
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        full_path = os.path.join(
            eval_directory, f"global_state_{file_name}_{formatted_datetime}.json"
        )

        with open(full_path, "w") as f:
            json.dump(env.global_state, f, cls=NumpyArrayEncoder)

    return outputs


def fetch_episode_states1(trainer_obj=None, episode_states=None, file_name=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    outputs = {}

    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    obs, _ = env_object.reset()

    env = env_object.env

    for state in episode_states:
        assert state in env.global_state, f"{state} is not in global state!"
        # Initialize the episode states
        array_shape = env.global_state[state]["value"].shape
        outputs[state] = np.nan * np.ones(array_shape)

    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_agents):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    for timestep in range(env.episode_length):
        for state in episode_states:
            outputs[state][timestep] = env.global_state[state]["value"][timestep]

        actions = {}
        # TODO: Consider using the `compute_actions` (instead of `compute_action`)
        # API below for speed-up when there are many agents.
        for region_id in range(env.num_agents):
            if (
                len(agent_states[region_id]) == 0
            ):  # stateless, with a linear model, for example

                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )

            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_actions(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
        obs, rewards, done, truncateds, info = env_object.step(actions)
        if done["__all__"]:
            for state in episode_states:
                outputs[state][timestep + 1] = env.global_state[state]["value"][
                    timestep + 1
                ]
            if file_name:
                # Get the current script's directory
                current_directory = os.path.dirname(__file__)
                # Construct the path to the 'eval' directory
                eval_directory = os.path.join(current_directory, "..", "evals")
                # Ensure the path is absolute
                eval_directory = os.path.abspath(eval_directory)
                formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
                name = f"global_state_{file_name}_{formatted_datetime}.json"
                # Define the file name and construct the full file path
                file_path = os.path.join(eval_directory, name)

                with open(file_path, "w") as f:
                    json.dump(env.global_state, f, cls=NumpyArrayEncoder)
            break

    return outputs


def set_num_agents(config_yaml):
    """
    updates the region_yamls folder with the appropriate number of regions
    as defined by the yaml config
    Purpose: ensures your environment has the correct YAML configs before training starts.
    """
    import shutil

    num_agents = config_yaml["env"]["regions"]["num_agents"]
    assert num_agents in [3, 7, 20, 27]

    # Updated paths
    target_directory = str(OTHER_YAMLS_DIR)
    if not os.path.exists(target_directory):
        raise FileNotFoundError(
            f"Expected other_yamls folder not found: {target_directory}"
        )

    folders = [
        entry
        for entry in os.listdir(target_directory)
        if os.path.isdir(os.path.join(target_directory, entry))
    ]

    target_region_yamls = os.path.join(
        target_directory,
        [folder for folder in folders if folder.startswith(str(num_agents))][0],
    )

    test_regions_directory = str(REGION_YAMLS_DIR)

    # Remove old region YAMLs (except default.yml)
    for file in os.listdir(test_regions_directory):
        file_path = os.path.join(test_regions_directory, file)
        if os.path.isfile(file_path) and file != "default.yml":
            os.remove(file_path)

    # Copy YAMLs from target_region_yamls
    for file in os.listdir(target_region_yamls):
        if file.endswith(".yml"):
            shutil.copy2(
                os.path.join(target_region_yamls, file),
                os.path.join(test_regions_directory, file),
            )

    # print(f"Region YAMLs updated to {num_agents} regions from {target_region_yamls}")


def discrete_action_to_float(index, num_bins=10, max_value=0.9):
    return (index / (num_bins - 1)) * max_value


# =========================
# Logging helpers
# =========================
def save_jsonl(save_dir, file_name, entries, suffix, append=False):
    """Save a list of entries into a JSONL file."""
    logs_dir = Path(save_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / f"{file_name}_{suffix}.jsonl"

    with open(jsonl_path, "a" if append else "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, cls=NumpyArrayEncoder) + "\n")

    print(f"✅ Saved {len(entries)} entries to JSONL at: {jsonl_path}")
    return str(jsonl_path)


def log_gradients_to_file(trainer, filepath="gradient_logs.txt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write("Some gradient info\n")
    print(f"[CALL CHECK] Called log_gradients_to_file with path: {filepath}")
    with open(filepath, "a") as f:
        f.write("Logging something...\n")
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print("\n🔧 DEBUG: Computing action head gradients...\n")
            for pid, policy in trainer.workers.local_worker().policy_map.items():
                model = policy.model
                print(f"🧠 Policy ID: {pid}")
                print(f"    Model type: {type(model)}")

                if hasattr(model, "compute_action_head_gradients"):
                    obs_space = policy.observation_space
                    sample_obs = obs_space.sample()
                    # If sample_obs is a dict, convert each part to tensor and batch dim
                    if isinstance(sample_obs, dict):
                        input_obs = {}
                        for k, v in sample_obs.items():
                            input_obs[k] = torch.tensor(
                                v, dtype=torch.float32
                            ).unsqueeze(0)
                    else:
                        input_obs = torch.tensor(
                            sample_obs, dtype=torch.float32
                        ).unsqueeze(0)

                    input_dict = {"obs": input_obs}

                    grad_info = model.compute_action_head_gradients(input_dict)
                    # print("    🎯 Gradient norms per action head:")
                    # for head_id, grads in grad_info:
                    #    print(f"      Head: {head_id}")
                    # for param, norm in grads.items():
                    #    print(f"        {param}: {norm:.6f}")
                else:
                    print(
                        "    🚫 Model does not implement compute_action_head_gradients."
                    )
        except Exception as e:
            print(f"❌ Exception in log_gradients_to_file: {e}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    # -------------------------
    # Setup run
    # -------------------------
    default_yaml = UTILS_DIR / "rice_rllib_discrete.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, default=default_yaml)
    args = parser.parse_args()

    # Load YAML configuration
    config_yaml = get_config_yaml(yaml_path=args.yaml)
    if config_yaml is None:
        raise ValueError(f"The run configuration is missing. Please check: {args.yaml}")

    # -------------------------
    # Checkpoints / climate saving config
    # -------------------------
    saving_cfg = config_yaml.get("saving", {})
    save_checkpoints_enabled = saving_cfg.get("save_checkpoints", True)
    save_climate_enabled = saving_cfg.get("save_climate_info", True)
    save_iter = saving_cfg.get("save_iteration", None)

    # -------------------------
    # Create save directory
    # -------------------------
    run_tag = str(int(time.time()))
    save_dir = create_save_dir_path(results_dir=run_tag)
    print(f"📁 Run folder created: {save_dir}")

    scenario_name = config_yaml["env"]["scenario"]
    num_agents = config_yaml["env"]["regions"]["num_agents"]
    file_name = f"{run_tag}_{scenario_name}_{num_agents}"
    print(f"📌 Run file_name prefix: {file_name}")

    # -------------------------
    # Initialize Ray
    # -------------------------
    ray.init(ignore_reinit_error=True)
    set_num_agents(config_yaml)

    # -------------------------
    # Create trainer
    # -------------------------
    trainer = create_trainer(config_yaml, save_dir, file_name)
    callback_instance = trainer.workers.local_worker().callbacks
    reward_hist_full = {}
    print(f"Number of policies: {len(trainer.workers.local_worker().policy_map)}")

    env_obj = trainer.workers.local_worker().env.env

    # ✅ Save BAU history to JSONL
    if hasattr(env_obj, "bau_history") and len(env_obj.bau_history) > 0:
        bau_file_path = save_jsonl(
            save_dir=save_dir,
            file_name=f"{file_name}_bau",
            entries=env_obj.bau_history,
            suffix="jsonl",
        )
        print(f"[DEBUG] BAU history saved to: {bau_file_path}")
    else:
        print("[DEBUG] No BAU history found to save.")

    env_wrapper = trainer.workers.local_worker().env
    episode_length = env_obj.episode_length

    num_episodes = config_yaml["trainer"]["num_episodes"]
    train_batch_size = config_yaml["trainer"]["train_batch_size"]

    episodes_per_iter = train_batch_size // episode_length
    num_iters = num_episodes // episodes_per_iter
    print(f"Episode length: {episode_length}")
    print(f"Total steps: {num_episodes * episode_length}")
    print(f"Episodes per iter: {episodes_per_iter}")

    learner_stats_logs = []  # Learner stats per iteration

    # -------------------------
    # Training loop
    # -------------------------
    for iteration in tqdm(range(num_iters), disable=True):
        if hasattr(callback_instance, "set_current_iteration"):
            callback_instance.set_current_iteration(iteration + 1)
        print(
            f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********",
            flush=True,
        )
        result = trainer.train()
        last_rewards = trainer.workers.local_worker().env.env.get_last_episode_rewards()
        for aid, r in last_rewards.items():
            reward_hist_full.setdefault(aid, []).append(r)
        # print("[DEBUG] Current reward history per agent:")
        # for aid, hist in callback_instance.reward_hist.items():
        #    print(f"  {aid}: {hist}")
        # -------------------------
        # Check for reward plateau (early stopping)
        # -------------------------
        moving_window = 20
        min_iters_before_check = 50
        consecutive_plateau_checks = 3
        plateau_counter = 0

        if iteration + 1 >= min_iters_before_check:
            stop_training = True
            for aid in reward_hist_full:
                if len(reward_hist_full[aid]) < 2 * moving_window:
                    stop_training = False
                    break

                # Compute adaptive threshold: 1% of recent mean absolute reward
                recent_abs_mean = np.mean(
                    np.abs(reward_hist_full[aid][-2 * moving_window :])
                )
                threshold = 0.01 * max(recent_abs_mean, 1e-6)  # avoid 0 threshold

                delta = np.abs(
                    np.mean(reward_hist_full[aid][-moving_window:])
                    - np.mean(
                        reward_hist_full[aid][-2 * moving_window : -moving_window]
                    )
                )

                if delta >= threshold:
                    stop_training = False
                    break

            if stop_training:
                plateau_counter += 1
                if plateau_counter >= consecutive_plateau_checks:
                    print(
                        f"[INFO] Stopping training: reward plateau reached at iteration {iteration+1}"
                    )
                    break
            else:
                plateau_counter = 0
    # -------------------------
    # Model checkpointing
    # -------------------------
    if save_checkpoints_enabled:
        total_timesteps = getattr(trainer, "num_env_steps_sampled", None)
        # or
        checkpoints_dir = save_dir / "checkpoints"
        save_model_checkpoint(trainer, checkpoints_dir, total_timesteps)

    if learner_stats_logs:
        save_jsonl(save_dir, file_name, learner_stats_logs, suffix="learner_stats")
    # Outside the training loop: zip everything once
    if save_checkpoints_enabled:
        zip_run_outputs(save_dir, file_name)

    print("[INFO] Training complete. Generating reward convergence plot...")
    print("[DEBUG] reward_hist_full before plotting:")
    for aid, hist in reward_hist_full.items():
        print(f"  {aid}: {hist}")

    plot_reward_convergence(reward_hist=reward_hist_full, save_dir=save_dir)
ray.shutdown()
