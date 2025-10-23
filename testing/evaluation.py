#!/usr/bin/env python3
"""
run this second!
for (FEATURES EVALUATION)
    python -m testing.evaluation --run_dir outputs/1761139151 --eval

    it will Look for outputs/1761139151/eval/episode_summaries.pkl
    Generate plots and metrics under outputs/1761139151/eval_analysis/

for (LEARNING EVALUATION)
    python -m testing.evaluation --run_dir outputs/1761139151 --learning

    Analyze training logs under outputs/1761139151/
    Save results to outputs/1761139151/learning_analysis/

Inputs required are:
1) episodes.pkl (which is episode_summaries.pkl)
2) A directory path (--eval_dir) where to read episodes.pkl and save plots/metrics

What should output:
1) Basic metrics printed in the console: Number of episodes, Average reward, Std, max, min reward
2) CSV summary:
    metrics_summary.csv in eval_dir containing basic stats.
3) Plots saved in eval_dir/plots/:
    rewards_<agent_id>.png for reward trajectories per agent
    <metric>.png for each metric in _METRICS_TO_LABEL_DICT, averaged over agents/regions
"""
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from testing.learning_eval import analyze_learning
from testing.features_eval import (
    load_episodes,
    compute_basic_metrics,
    plot_rewards,
    plot_info_variables,
    plot_metrics_per_region,
    plot_global_variables_combined,
)


# ----------------------------
# Utility functions
# ----------------------------
def find_and_load_config(run_dir: Path):
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML file found in {run_dir}")
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# ----------------------------
# CLI entrypoint
# ----------------------------


def main():
    "Start Evaluation"
    parser = argparse.ArgumentParser(
        description="Analyze RICE-N training and evaluation results."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to the training run folder."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation analysis from episodes.pkl"
    )
    parser.add_argument(
        "--learning",
        action="store_true",
        help="Run learning analysis from training logs",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run only evaluation analysis (skip learning analysis)",
    )
    parser.add_argument(
        "--episodes_file",
        type=str,
        default="episode_summaries.pkl",
        help="Name of episodes pickle file in eval dir",
    )
    parser.add_argument(
        "--feat_out",
        type=str,
        default=None,
        help="Folder name for evaluation output of features (defaults to eval_analysis)",
    )
    parser.add_argument(
        "--learning_out",
        type=str,
        default=None,
        help="Folder name for learning output (defaults to learning_analysis)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # -----------------------
    # Evaluation analysis
    # -----------------------

    eval_base = run_dir / "eval"
    feat_out = eval_base / "feat_dir"
    config = find_and_load_config(eval_base)
    num_agents = config.get("env", {}).get("regions", {}).get("num_agents", None)
    print("num_agents", num_agents)

    if not eval_base.exists():
        print(f"⚠️ No eval folder found at {eval_base}, skipping evaluation analysis.")
    else:
        # Ensure feat_dir exists
        feat_out.mkdir(parents=True, exist_ok=True)
        print(f"\n📊 Running feature analysis → {feat_out}")

        try:

            episodes = load_episodes(eval_base, episodes_file=args.episodes_file)
            metrics = compute_basic_metrics(episodes)
            df = pd.DataFrame([metrics])
            df.to_csv(feat_out / "metrics_summary.csv", index=False)
            print(f"💾 Metrics saved to {feat_out / 'metrics_summary.csv'}")

            plot_rewards(episodes, feat_out)
            plot_info_variables(episodes, feat_out)
            plot_global_variables_combined(episodes, feat_out)
            if num_agents:
                plot_metrics_per_region(
                    episodes, feat_out, regions=list(range(num_agents))
                )
            print(f"✅ Evaluation plots saved → {feat_out}")
        except Exception as e:
            print(f"⚠️ Evaluation analysis failed: {e}")


if __name__ == "__main__":
    main()
