"""
Evaluation script for RLlib multi-agent environment using saved checkpoints.
Run it from the repo root (RICE-N-exp)
run
python -m testing.test_eval --run outputs/1760696745/1760696745_default_7.zip --episodes 50
50 tells how many episodes to run in the eval

"""

import os
import argparse
import shutil
import json
import yaml
import numpy as np
import logging
from pathlib import Path
import tempfile
import zipfile
import torch
from scipy import stats
import matplotlib.pyplot as plt

from my_project.utils.fixed_paths import SCRIPTS_DIR
import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from trainer import create_trainer, load_model_checkpoints, save_jsonl
import logging

logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------------------------------------------------------------


def load_run_config(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def prepare_eval_trainer(config_yaml: dict, run_dir: str, seed: int = None):
    """
    Create RLlib trainer for evaluation with multi-agent checkpoint loading.
    """
    config_yaml = config_yaml.copy()
    config_yaml["trainer"]["num_workers"] = 0
    if seed is not None:
        config_yaml["seed"] = seed

    trainer = create_trainer(config_yaml, save_dir=run_dir)

    checkpoints_dir = Path(run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        load_model_checkpoints(trainer, checkpoints_dir, ckpt_idx=-1)
    return trainer


def rollout_episodes(trainer, num_episodes, eval_dir=None):
    results = []
    for _ in range(num_episodes):
        episode_result = trainer.workers.local_worker().sample()
        results.append(episode_result)
        # Example: optionally save to eval_dir
        if eval_dir is not None:
            with open(eval_dir / "episode_log.txt", "a") as f:
                f.write(str(episode_result) + "\n")
    return results


def copy_reward_convergence(run_dir: Path, eval_dir: Path):
    """
    Copy reward convergence plot from training folder to eval/plots/ if it exists.
    """
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Typically saved as reward_convergence.png during training
    src_plot = run_dir / "reward_convergence.png"
    if src_plot.exists():
        shutil.copy(src_plot, plots_dir / "reward_convergence.png")
        logging.info(f"Copied reward convergence plot to {plots_dir}")
    else:
        logging.warning(f"No reward_convergence.png found in {run_dir}")


def evaluate_run(run_dir_or_zip: str, num_episodes: int = 100, eval_seed: int = None):
    run_path = Path(run_dir_or_zip)

    # Determine base run folder
    if run_path.suffix == ".zip":
        tmp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(run_path, "r") as zf:
            zf.extractall(tmp_dir.name)
        run_dir = Path(tmp_dir.name)
        parent_run_id = run_path.stem.split("_")[0]
        run_base_dir = Path(run_path.parent) / parent_run_id
    else:
        run_dir = run_path
        run_base_dir = run_dir

    # Create dedicated eval folder
    eval_dir = run_base_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load config and prepare trainer
    config_yaml = load_run_config(run_dir)
    trainer = prepare_eval_trainer(config_yaml, run_dir, seed=eval_seed)

    # Unwrap the environment completely
    env = trainer.workers.local_worker().env
    while hasattr(env, "env"):
        env = env.env

    # Suppress logging if possible
    if getattr(env, "logger", None) is not None:
        env.logger.setLevel(logging.WARNING)
    elif hasattr(env, "verbose"):
        env.verbose = 0

    # Run episodes
    results = rollout_episodes(trainer, num_episodes, eval_dir=eval_dir)

    # Copy reward convergence files (if any)
    copy_reward_convergence(run_dir, eval_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved RLlib run.")
    parser.add_argument(
        "--run", type=str, required=True, help="Path to run folder or zip file"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for evaluation"
    )
    args = parser.parse_args()

    results = evaluate_run(
        run_dir_or_zip=args.run, num_episodes=args.episodes, eval_seed=args.seed
    )
    print("Evaluation complete. Sample of results:")
    for k, v in list(results.items())[:2]:
        print(f"{k}: {v}")
