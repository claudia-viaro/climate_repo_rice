# =====================================================
# === LEARNING (TRAINING) ANALYSIS AND VISUALIZATION ===
# =====================================================

"""
python -m testing.learning_eval --run_dir outputs/1761147954

Analyze training logs under outputs/1761139151/
Save results to outputs/1761139151/learning_analysis/

"""

import json
import glob
import yaml
import argparse
import pickle
import tempfile
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict


# -------------------------------
# Extract logs from zip or folder
# -------------------------------
def extract_logs(run_dir_or_zip: str):
    run_path = Path(run_dir_or_zip)
    tmp_dir = None

    # if folder, check for zip inside
    if run_path.is_dir():
        zip_files = list(run_path.glob("*.zip"))
        if zip_files:
            run_path = zip_files[0]

    if run_path.suffix == ".zip":
        tmp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(run_path, "r") as zf:
            zf.extractall(tmp_dir.name)
        extracted_dir = Path(tmp_dir.name)
        logs_dir = next(extracted_dir.rglob("logs"), None)
        if logs_dir is None:
            raise FileNotFoundError(f"No logs folder found inside zip: {run_path}")
        return logs_dir, tmp_dir

    else:
        logs_dir = next(run_path.rglob("logs"), None)
        if logs_dir is None:
            raise FileNotFoundError(f"No logs folder found in folder: {run_path}")
        return logs_dir, None


# -------------------------------
# Load JSONL logs
# -------------------------------
# -------------------------------
# Load JSONL logs
# -------------------------------
def load_training_logs(logs_dir: Path) -> dict:
    def _load_jsonl(file_path):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)

    logs = {}

    # Aggregated learner stats
    agg_files = list(logs_dir.glob("*learner_stats_aggregated.jsonl"))
    if agg_files:
        logs["aggregated"] = _load_jsonl(agg_files[-1])
        print(f"✅ Loaded aggregated stats: {agg_files[-1].name}")

    # Per-policy stats
    policy_files = list(logs_dir.glob("*learner_stats_per_policy.jsonl"))
    if policy_files:
        logs["per_policy"] = _load_jsonl(policy_files[-1])
        print(f"✅ Loaded per-policy stats: {policy_files[-1].name}")

    # Per-agent rewards
    agent_files = list(logs_dir.glob("*per_agent_rewards.jsonl"))
    if agent_files:
        logs["per_agent"] = _load_jsonl(agent_files[-1])
        print(f"✅ Loaded per-agent rewards: {agent_files[-1].name}")

    return logs


# -----------------------------------------------------
# Plot learning and convergence diagnostics
# -----------------------------------------------------
# -------------------------------
# Plot metrics and extra metrics
# -------------------------------
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def plot_learning_curves(logs: dict, save_dir: Path, smooth_window: int = 5):
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    # ----------------- Aggregated metrics -----------------
    if "aggregated" in logs:
        df = logs["aggregated"].sort_values("iteration")
        df["iteration"] = df["iteration"].astype(int)

        metrics_to_plot = [
            ("avg_reward", "Average Reward", True),
            ("policy_loss", "Policy Loss", False),
            ("vf_loss", "Value Function Loss", False),
            ("entropy", "Policy Entropy", False),
            ("kl", "KL Divergence", False),
            ("vf_explained_var", "Value Function Explained Var", False),
        ]

        for col, title, smooth in metrics_to_plot:
            if col not in df.columns:
                continue

            plt.figure(figsize=(6, 4))
            y = (
                df[col].rolling(smooth_window, min_periods=1).mean()
                if smooth
                else df[col]
            )
            plt.plot(df["iteration"], y, "k-", linewidth=1.8)
            plt.title(title)
            if col == "avg_reward":
                plt.suptitle("Averaged over agents per iteration", fontsize=9)
            plt.xlabel("Iteration")
            plt.ylabel(title)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f"{col}.pdf", dpi=150)
            plt.close()

            summary_rows.append(
                {
                    "metric": col,
                    "mean": np.nanmean(df[col]),
                    "std": np.nanstd(df[col]),
                    "last": df[col].iloc[-1],
                }
            )

        print(f"✅ Aggregated metric plots saved to {save_dir}")

    # ----------------- Per-policy metrics -----------------
    if "per_policy" in logs:
        dfp = logs["per_policy"]
        if {"iteration", "policy_id"}.issubset(dfp.columns):
            grouped = dfp.groupby(["policy_id", "iteration"]).mean(numeric_only=True)
            for pid in grouped.index.get_level_values(0).unique():
                subset = grouped.loc[pid]

                # Plot policy_loss & vf_loss together
                plt.figure(figsize=(6, 4))
                plt.plot(subset.index, subset["policy_loss"], label="Policy Loss")
                plt.plot(subset.index, subset["vf_loss"], label="Value Loss")
                plt.title(f"Policy {pid} Loss Metrics")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                ax = plt.gca()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(save_dir / f"policy_{pid}_loss.pdf", dpi=150)
                plt.close()

            # KL divergence across policies (optional)
            if "kl" in dfp.columns:
                policy_var = dfp.groupby("iteration")["kl"].var()
                policy_var_smoothed = policy_var.rolling(
                    smooth_window, min_periods=1
                ).mean()

                plt.figure(figsize=(6, 4))
                plt.plot(policy_var_smoothed.index, policy_var_smoothed.values, "r-")
                plt.title("Smoothed KL Variance Across Policies")
                plt.xlabel("Iteration")
                plt.ylabel("KL Variance")
                ax = plt.gca()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(save_dir / "kl_variance.pdf", dpi=150)
                plt.close()
                print("📊 Per-policy metrics and KL variance plots done.")

    # ----------------- Per-agent rewards -----------------
    if "per_agent" in logs:
        dfr = logs["per_agent"]
        if {"iteration", "agent_id", "reward"}.issubset(dfr.columns):
            plt.figure(figsize=(7, 5))
            cumulative_rewards = {}

            for idx, (aid, subdf) in enumerate(dfr.groupby("agent_id")):
                subdf = subdf.sort_values("iteration")
                smoothed = subdf["reward"].rolling(smooth_window, min_periods=1).mean()
                plt.plot(subdf["iteration"], smoothed, label=f"R {idx}", alpha=0.7)
                cumulative_rewards[aid] = subdf["reward"].sum()

            plt.title("Per-Agent Reward Convergence (Smoothed)")
            plt.xlabel("Iteration")
            plt.ylabel("Reward")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "per_agent_reward_convergence.pdf", dpi=150)
            plt.close()

            # Plot cumulative reward per agent
            plt.figure(figsize=(6, 4))
            agents = list(cumulative_rewards.keys())
            plt.bar(range(len(agents)), cumulative_rewards.values(), color="c")
            plt.xticks(range(len(agents)), [f"R {i}" for i in range(len(agents))])
            plt.title("Cumulative Reward per Agent")
            plt.xlabel("")
            plt.ylabel("Total Reward")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(save_dir / "per_agent_cumulative_reward.pdf", dpi=150)
            plt.close()

            # Smoothed reward variance across agents
            reward_var = dfr.groupby("iteration")["reward"].var()
            reward_var_smoothed = reward_var.rolling(
                smooth_window, min_periods=1
            ).mean()

            plt.figure(figsize=(6, 4))
            plt.plot(reward_var_smoothed.index, reward_var_smoothed.values, "m-")
            plt.title("Smoothed Reward Variance Across Agents")
            plt.xlabel("Iteration")
            plt.ylabel("Reward Variance")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "reward_variance.pdf", dpi=150)
            plt.close()

            # Summary CSV
            summary_df = (
                dfr.groupby("agent_id")["reward"]
                .agg(["mean", "std", "sum"])
                .reset_index()
            )
            summary_df.rename(columns={"sum": "cumulative"}, inplace=True)
            summary_df.to_csv(save_dir / "per_agent_reward_summary.csv", index=False)
            print(
                f"💾 Per-agent reward summary saved → {save_dir/'per_agent_reward_summary.csv'}"
            )

    # ----------------- Save combined summary -----------------
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            save_dir / "learning_summary.csv", index=False
        )
        print(f"💾 Combined learning summary saved → {save_dir/'learning_summary.csv'}")

    print(f"✅ All learning plots and metrics saved to {save_dir}")


# =====================================================
# === WRAPPER ===
# =====================================================
def analyze_learning_run(run_dir_or_zip):
    logs_dir, tmp_dir = extract_logs(run_dir_or_zip)
    # determine correct save path
    base_dir = (
        Path(run_dir_or_zip).parent
        if Path(run_dir_or_zip).suffix == ".zip"
        else Path(run_dir_or_zip)
    )
    save_dir = base_dir / "eval" / "learn_dict"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs = load_training_logs(logs_dir)
    plot_learning_curves(logs, save_dir)
    if tmp_dir:
        tmp_dir.cleanup()
    print(f"✅ Analysis complete. Results saved to {save_dir}")


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze RICE-N training results.")
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to run folder or zip file"
    )
    args = parser.parse_args()
    analyze_learning_run(args.run_dir)


if __name__ == "__main__":
    main()
