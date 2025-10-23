"""
run this first!

Evaluation script for RLlib multi-agent environment using saved checkpoints.
Run it from the repo root (RICE-N-exp)
run
python -m testing.test_eval --run outputs/1760881212/1760881212_default_7.zip --episodes 50
50 tells how many episodes to run in the eval

1) Loads the same trained policies
2) Steps through your environment exactly as RLlib does
3) Saves the same structure (episodes.pkl)
4) Avoids any extra Weights & Biases logging
5) Works both with zipped runs or unzipped directories

It produce: 50 episodes
1) Results: outputs/1760881212/eval/episode_summaries.pkl

episode_summaries.pkl
    List of all episodes, each entry contains:

    {
        "episode_idx": 0,
        "timesteps": 20,
        "total_reward": {
            0: 1.23,
            1: 1.15,
            2: 1.20
        },
        "climate_info": {
            "timestep": 20,
            "rl_summary": [...],  # 20 timesteps of RL run data
            "bau_history": [...]  # 20 timesteps of baseline data
        }
    }
"""

import os
import ray
from datetime import datetime

import argparse
import shutil
import json
import yaml
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import pickle
import tempfile
import zipfile
import torch
from scipy import stats
import matplotlib.pyplot as plt

from ray.rllib.algorithms import Algorithm

# from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.sample_batch import SampleBatch

from my_project.utils.fixed_paths import SCRIPTS_DIR

import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from trainer import create_trainer, load_model_checkpoints, save_jsonl

logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------
# Load run configuration
# ---------------------------------------------------------------------
def load_run_config(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------
# Prepare trainer for evaluation
# ---------------------------------------------------------------------
def prepare_eval_trainer(
    config_yaml: dict, run_dir: str, seed: int = None
) -> Algorithm:
    config_yaml = config_yaml.copy()
    config_yaml["trainer"]["num_workers"] = 0
    if seed is not None:
        config_yaml["seed"] = seed

    trainer = create_trainer(config_yaml, save_dir=run_dir)
    checkpoints_dir = Path(run_dir) / "checkpoints"

    if checkpoints_dir.exists():
        load_model_checkpoints(trainer, checkpoints_dir, ckpt_idx=-1)
    else:
        print(f"[WARN] No checkpoints found in {checkpoints_dir}")

    return trainer


# ---------------------------------------------------------------------
# Rollout true episodes
# ---------------------------------------------------------------------
def rollout_true_episodes(trainer: Algorithm, num_episodes: int, eval_dir: Path):
    results = []

    # Unwrap environment to base Rice
    env = trainer.workers.local_worker().env
    while hasattr(env, "env"):
        env = env.env
    if not hasattr(env, "bau_history"):
        print("⚠️ env has no bau_history, BAU curves will be empty")
    if not hasattr(env, "last_info_rl"):
        print("⚠️ env has no last_info_rl, experience curves will be empty")

    print(f"[INFO] Using environment: {env.__class__.__name__}")

    for ep_idx in tqdm(range(num_episodes), desc="Evaluating episodes"):
        obs, _ = env.reset()
        done = {"__all__": False}
        t = 0

        # Storage for per-agent rewards
        ep_rewards = {a: [] for a in obs.keys()}

        last_climate_info = None

        while not done["__all__"]:
            # Compute actions for all agents
            actions = {
                agent_id: trainer.compute_single_action(
                    o,
                    policy_id=trainer.workers.local_worker().policy_mapping_fn(
                        agent_id
                    ),
                )
                for agent_id, o in obs.items()
            }

            next_obs, rewards, done, _, infos = env.step(actions)

            for agent_id, r in rewards.items():
                ep_rewards[agent_id].append(r)

            # Capture final climate step info
            common_info = infos.get("__common__", {})
            if common_info.get("stage") == "climate":
                last_climate_info = {"timestep": common_info.get("timestep")}

                # RL summary collected by wrapper at final step
                if hasattr(env, "last_info_rl"):
                    last_climate_info["rl_summary"] = env.last_info_rl

                # Full BAU trajectory
                if hasattr(env, "bau_history"):
                    last_climate_info["bau_history"] = env.bau_history

            obs = next_obs
            t += 1

        # Store episode-level summary
        episode_summary = {
            "episode_idx": ep_idx,
            "timesteps": t,
            "total_reward": {a: float(np.sum(r)) for a, r in ep_rewards.items()},
            "climate_info": last_climate_info,
        }
        results.append(episode_summary)

        print(
            f"✅ Episode {ep_idx} done. Climate info captured: {last_climate_info is not None}"
        )

    # Save episodes and climate info separately
    save_path = eval_dir / "episode_summaries.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Saved {len(results)} episodes")

    return results


# ---------------------------------------------------------------------
# Copy all plot produced during training
# ---------------------------------------------------------------------
def copy_all_plots(run_dir: Path, eval_dir: Path):
    """
    Copies all PNG plots generated by training (reward/loss curves/etc.)
    from the run directory to the evaluation folder.
    """
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNG plots in the run_dir
    plot_files = list(run_dir.glob("*.png"))
    if not plot_files:
        print(f"[WARN] No plot files found in {run_dir}")
        return

    for src_plot in plot_files:
        dst = plots_dir / src_plot.name
        shutil.copy(src_plot, dst)
        print(f"[INFO] Copied {src_plot.name} to {plots_dir}")

    print(f"✅ All available plots copied to {plots_dir}")


# ---------------------------------------------------------------------
# Evaluate a run
# ---------------------------------------------------------------------
def evaluate_run(run_dir_or_zip: str, num_episodes: int = 5, eval_seed: int = None):
    run_path = Path(run_dir_or_zip)
    tmp_dir = None  # Keep reference so it doesn’t get deleted too early

    # ---------------------------
    # 1. Unzip or load directory
    # ---------------------------
    if run_path.suffix == ".zip":
        tmp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(run_path, "r") as zf:
            zf.extractall(tmp_dir.name)
        run_dir = Path(tmp_dir.name)
        run_base_dir = Path(run_path.parent)
    else:
        run_dir = run_path
        run_base_dir = run_path

    # ---------------------------
    # 2. Create unique eval folder
    # ---------------------------
    eval_dir = run_base_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 3. Move YAML to eval folder
    # ---------------------------
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        # Sometimes the YAML may be nested under a subfolder (like the zip structure)
        yaml_files = list(run_dir.glob("**/*.yaml"))
    if yaml_files:
        yaml_src = yaml_files[0]
        yaml_dst = eval_dir / yaml_src.name
        shutil.copy(yaml_src, yaml_dst)
        print(f"[INFO] Copied YAML config to {yaml_dst}")
    else:
        print("[WARN] No YAML config found in the unzipped run.")

    # ---------------------------
    # 4. Load config and trainer
    # ---------------------------
    config_yaml = load_run_config(run_dir)
    trainer = prepare_eval_trainer(config_yaml, run_dir, seed=eval_seed)

    # ---------------------------
    # 5. Create extra folders based on config flags
    # ---------------------------
    save_cfg = config_yaml.get("saving", {})
    save_checkpoints = save_cfg.get("save_checkpoints", False)
    save_policy_enabled = save_cfg.get("save_policy_info", False)

    if save_checkpoints:
        feat_dir = eval_dir / "feat_dir"
        feat_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created feature evaluation folder: {feat_dir}")

    if save_policy_enabled:
        learn_dir = eval_dir / "learn_dir"
        learn_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created learning evaluation folder: {learn_dir}")

    # ---------------------------
    # 6. Run evaluation rollout
    # ---------------------------
    results = rollout_true_episodes(trainer, num_episodes, eval_dir)

    # ---------------------------
    # 7. Copy reward convergence plot
    # ---------------------------
    copy_all_plots(run_dir, eval_dir)

    # ---------------------------
    # 8. Clean up Ray and temp dir
    # ---------------------------
    ray.shutdown()
    if tmp_dir is not None:
        tmp_dir.cleanup()

    return results, eval_dir


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved RLlib run.")
    parser.add_argument(
        "--run", type=str, required=True, help="Path to run folder or zip file"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results, eval_dir = evaluate_run(
        args.run, num_episodes=args.episodes, eval_seed=args.seed
    )
    print(f"\n✅ Evaluation complete. Results saved under: {eval_dir}")
