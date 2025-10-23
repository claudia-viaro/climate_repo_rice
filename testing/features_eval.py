import yaml
import argparse
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# ----------------------------
# Metric label configuration
# ----------------------------

_METRICS_TO_LABEL_DICT = OrderedDict(
    [
        ("global_temperature", ("Temperature Rise (°C)", 2)),
        ("global_carbon_mass", ("Carbon Mass (GtC)", 2)),
        ("global_emissions", ("Global Emissions (GtCO₂/yr)", 2)),
        ("aux_m_all_regions", ("Regional Carbon Mass", 2)),
        ("capital_all_regions", ("Capital", 2)),
        ("labor_all_regions", ("Labor", 2)),
        ("production_factor_all_regions", ("Production Factor", 2)),
        ("production_all_regions", ("Production", 2)),
        ("intensity_all_regions", ("Intensity", 2)),
        ("savings_all_regions", ("Savings", 2)),
        ("mitigation_rates_all_regions", ("Mitigation Rate", 2)),
        ("mitigation_cost_all_regions", ("Mitigation Cost", 2)),
        ("damages_all_regions", ("Damages", 2)),
        ("abatement_cost_all_regions", ("Abatement Cost", 2)),
        ("utility_all_regions", ("Utility", 2)),
        ("social_welfare_all_regions", ("Social Welfare", 2)),
        ("reward_all_regions", ("Reward", 2)),
        ("current_balance_all_regions", ("Current Balance", 2)),
        ("gross_output_all_regions", ("Gross Output", 2)),
        ("investment_all_regions", ("Investment", 2)),
        ("minimum_mitigation_rate_all_regions", ("Min Mitigation Rate", 2)),
        ("promised_mitigation_rate", ("Promised Mitigation Rate", 2)),
        ("requested_mitigation_rate", ("Requested Mitigation Rate", 2)),
        ("proposal_decisions", ("Proposal Decisions", 2)),
    ]
)


def get_years(num_steps=20, start_year=2015, step=5):
    return np.arange(start_year, start_year + num_steps * step, step)


def load_episodes(eval_dir: Path, episodes_file="episodes.pkl"):
    paths_to_try = [
        eval_dir / episodes_file,
        eval_dir / "episode_summaries.pkl",
    ]
    for path in paths_to_try:
        if path.exists():
            with open(path, "rb") as f:
                episodes = pickle.load(f)
            print(f"📂 Loaded {len(episodes)} episodes from {path}")
            return episodes
    raise FileNotFoundError(f"No episodes file found in {eval_dir}")


def average_over_regions(value):
    """Average region values if multi-region, otherwise return scalar."""
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value, dtype=float)
        if arr.ndim == 1 and arr.size > 1:
            return np.nanmean(arr)
        elif arr.ndim == 0:
            return float(arr)
    elif isinstance(value, (float, int)):
        return value
    return np.nan


def compute_basic_metrics(episodes):
    """Compute simple aggregate reward statistics across all episodes."""
    all_rewards = []
    for ep in episodes:
        rl_summary = ep["climate_info"]["rl_summary"]
        for step in rl_summary:
            reward = step["reward_all_regions"]
            all_rewards.extend(reward)
    metrics = {
        "num_episodes": len(episodes),
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
    }
    return metrics


# ----------------------------
# Plotting functions
# ----------------------------


def plot_rewards(episodes, eval_dir: Path):
    """
    Plot RL rewards averaged across regions, with mean ± std across episodes.
    x-axis shows years (2015 → ...).
    """
    eval_dir.mkdir(exist_ok=True)

    reward_curves = []
    for ep in episodes:
        climate_info = ep.get("climate_info", {})
        rl_summary = climate_info.get("rl_summary", [])
        timestep_rewards = [
            np.nanmean(step.get("reward_all_regions", np.nan)) for step in rl_summary
        ]
        if not np.all(np.isnan(timestep_rewards)):
            reward_curves.append(timestep_rewards)

    if not reward_curves:
        print("⚠️ No reward trajectories found; skipping reward plot.")
        return

    reward_curves = np.array(reward_curves)
    timesteps = reward_curves.shape[1]
    years = get_years(num_steps=timesteps)

    rl_mean = np.nanmean(reward_curves, axis=0)
    rl_std = np.nanstd(reward_curves, axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(years, rl_mean, color="blue", linewidth=2, label="RL mean reward")
    plt.fill_between(years, rl_mean - rl_std, rl_mean + rl_std, color="blue", alpha=0.2)
    for vals in reward_curves:
        plt.plot(years, vals, color="blue", alpha=0.1)

    plt.xlabel("Year")
    plt.ylabel("Reward")
    plt.title("RL Reward (Mean ± Std Across 50 Episodes)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "rewards.pdf", dpi=150)
    plt.close()

    print(f"✅ Reward mean ± std plot saved to {eval_dir / 'rewards.pdf'}")


def plot_global_variables_combined(episodes, eval_dir: Path):
    """
    Plot key global variables in a combined 2x3 grid:
    Top row: RL median ± CI + BAU
    Bottom row: RL median ± CI only
    """
    eval_dir.mkdir(exist_ok=True)

    variables = [
        "global_temperature",
        "global_emissions",
        "damages_all_regions",
    ]
    pretty_names = {
        "global_temperature": "Global Temperature (°C)",
        "global_emissions": "Global Emissions (GtCO₂/yr)",
        "damages_all_regions": "Damages (fraction of GDP)",
    }

    # Prepare RL and BAU time series per episode
    rl_data = {v: [] for v in variables}
    bau_data = {v: [] for v in variables}

    for ep in episodes:
        ci = ep.get("climate_info", {})
        rl_summary = ci.get("rl_summary", [])
        bau_history = ci.get("bau_history", [])

        for var in variables:
            rl_vals = [
                np.nanmean(np.array(step.get(var, np.nan))) for step in rl_summary
            ]
            rl_data[var].append(rl_vals)

            if bau_history:
                bau_vals = [
                    np.nanmean(np.array(step.get(var, np.nan))) for step in bau_history
                ]
                bau_data[var].append(bau_vals)

    # Convert to arrays and compute percentiles
    agg = {}
    for var in variables:
        arr = np.array(rl_data[var], dtype=float)
        agg[var] = {
            "median": np.nanmedian(arr, axis=0),
            "lower": np.nanpercentile(arr, 5, axis=0),
            "upper": np.nanpercentile(arr, 95, axis=0),
        }
        if bau_data[var]:
            bau_arr = np.array(bau_data[var], dtype=float)
            agg[var]["bau"] = np.nanmean(bau_arr, axis=0)
        else:
            agg[var]["bau"] = None

    # Years
    num_steps = len(next(iter(agg.values()))["median"])
    years = np.arange(2015, 2015 + 5 * num_steps, 5)

    # --- Plotting ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for i, var in enumerate(variables):
        data = agg[var]
        # --- Top row: RL + BAU
        ax_top = axes[i]
        ax_top.plot(years, data["median"], color="blue", linewidth=2, label="RL median")
        ax_top.fill_between(
            years,
            data["lower"],
            data["upper"],
            color="blue",
            alpha=0.2,
            label="RL 5–95% CI",
        )
        if data["bau"] is not None:
            ax_top.plot(
                years,
                data["bau"],
                color="orange",
                linestyle="--",
                linewidth=2,
                label="BAU",
            )
        ax_top.set_title(pretty_names[var] + " + BAU")
        ax_top.set_xlabel("Year")
        ax_top.set_ylabel(pretty_names[var])
        ax_top.legend()

        # --- Bottom row: RL only
        ax_bottom = axes[i + 3]
        ax_bottom.plot(years, data["median"], color="blue", linewidth=2)
        ax_bottom.fill_between(
            years, data["lower"], data["upper"], color="blue", alpha=0.2
        )
        ax_bottom.set_title(pretty_names[var] + " (RL only)")
        ax_bottom.set_xlabel("Year")
        ax_bottom.set_ylabel(pretty_names[var])
        ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = eval_dir / "global_variables_combined.pdf"
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"✅ Combined 2x3 global variable plot saved to {output_file}")


def plot_metrics_with_bau(episodes, eval_dir: Path):
    """
    For each metric in _METRICS_TO_LABEL_DICT:
    - RL curve = mean ± std across episodes (averaged across regions)
    - BAU curve = mean across timesteps (if available)
    - Save CSV summary for all metrics
    """
    eval_dir.mkdir(exist_ok=True)

    # Prepare CSV summary
    summary_rows = []

    # Sample shape
    climate_info = episodes[0]["climate_info"]
    rl_summary_len = len(climate_info["rl_summary"])
    years = get_years(num_steps=rl_summary_len)

    for metric, (label, _) in _METRICS_TO_LABEL_DICT.items():
        rl_values = []
        bau_values = []

        for ep in episodes:
            ci = ep["climate_info"]
            rl_summary = ci.get("rl_summary", [])
            bau_history = ci.get("bau_history", [])

            # RL per timestep (average across regions)
            rl_vals = [np.nanmean(step.get(metric, np.nan)) for step in rl_summary]
            if not np.all(np.isnan(rl_vals)):
                rl_values.append(rl_vals)

            # BAU per timestep (only one per episode — usually identical)
            if bau_history:
                bau_vals = [
                    np.nanmean(step.get(metric, np.nan)) for step in bau_history
                ]
                bau_values.append(bau_vals)

        if not rl_values:
            continue

        rl_values = np.array(rl_values)
        rl_mean = np.nanmean(rl_values, axis=0)
        rl_std = np.nanstd(rl_values, axis=0)

        if bau_values:
            bau_mean = np.nanmean(np.array(bau_values), axis=0)
        else:
            bau_mean = [np.nan] * len(years)

        # Save rows for CSV
        for yr, rl_m, rl_s, bau_m in zip(years, rl_mean, rl_std, bau_mean):
            summary_rows.append(
                {
                    "metric": metric,
                    "year": yr,
                    "rl_mean": rl_m,
                    "rl_std": rl_s,
                    "bau_mean": bau_m,
                }
            )

        # Plot
        plt.figure(figsize=(7, 4))
        plt.plot(years, rl_mean, color="blue", linewidth=2, label="RL mean ± std")
        plt.fill_between(
            years, rl_mean - rl_std, rl_mean + rl_std, color="blue", alpha=0.2
        )

        if bau_values:
            plt.plot(
                years,
                bau_mean,
                color="orange",
                linewidth=2,
                linestyle="--",
                label="BAU",
            )

        plt.xlabel("Year")
        plt.ylabel(label)
        plt.title(label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(eval_dir / f"{metric}.pdf", dpi=150)
        plt.close()

    # Save CSV summary
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(eval_dir / "metrics_summary.csv", index=False)
    print(f"✅ Metric plots and CSV saved to {eval_dir} (metrics_summary.csv)")


def plot_metrics_per_region(episodes, eval_dir: Path, regions=None):
    """
    Plot RL metrics per region (one line per region, no BAU).
    Handles scalar and array metrics safely, only plots metrics defined in _METRICS_TO_LABEL_DICT.
    """
    eval_dir.mkdir(exist_ok=True)

    for metric, (label, _) in _METRICS_TO_LABEL_DICT.items():
        # Check if metric exists in first episode's first timestep
        first_step = episodes[0]["climate_info"]["rl_summary"][0].get(metric)
        if first_step is None:
            continue

        # Determine number of regions
        if isinstance(first_step, (list, np.ndarray)):
            n_regions = len(first_step)
        else:
            n_regions = 1

        # Determine which regions to plot
        if regions is None:
            selected_regions = list(range(n_regions))
        else:
            selected_regions = [r for r in regions if r < n_regions]
        if not selected_regions:
            continue

        plt.figure(figsize=(6, 4))

        for r in selected_regions:
            region_curves = []

            for ep in episodes:
                rl_summary = ep["climate_info"]["rl_summary"]
                vals = []
                for step in rl_summary:
                    val = step.get(metric, np.nan)
                    if isinstance(val, (list, np.ndarray)):
                        vals.append(val[r] if r < len(val) else np.nan)
                    else:
                        # scalar metrics only plotted for region 0
                        if r == 0:
                            vals.append(val)
                        else:
                            vals.append(np.nan)
                region_curves.append(vals)

            arr = np.array(region_curves, dtype=float)
            mean_curve = np.nanmean(arr, axis=0)
            std_curve = np.nanstd(arr, axis=0)

            timesteps = np.arange(len(mean_curve))
            years = 2015 + timesteps * 5
            plt.plot(years, mean_curve, label=f"R{r}")
            plt.fill_between(
                years, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2
            )

        plt.title(f"{label} per Region")
        plt.xlabel("Year")
        plt.ylabel(label)
        plt.xticks(years)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(eval_dir / f"{metric}_per_region.pdf", dpi=150)
        plt.close()

    print(f"✅ Per-region metric plots saved under {eval_dir}")


def plot_info_variables(episodes, eval_dir: Path):
    """
    Plot all RL features stored in climate_info['rl_summary'] across episodes.
    Each feature gets a mean ± episode range plot.
    eval_dir.mkdir(exist_ok=True)
    """
    eval_dir.mkdir(exist_ok=True)

    # --- Determine sample keys from first episode ---
    # Assumes each episode is a dict with agent IDs as keys
    sample_info = episodes[0][list(episodes[0].keys())[0]]["infos"]
    if not isinstance(sample_info, (list, np.ndarray)) or not isinstance(
        sample_info[0], dict
    ):
        print("⚠️ No info dicts found in episodes — skipping info variable plots.")
        return

    # Iterate through metrics defined in _METRICS_TO_LABEL_DICT
    for metric, (label, round_decimals) in _METRICS_TO_LABEL_DICT.items():
        all_values = []

        for ep in episodes:
            # ep is a dict of agent data
            for agent_data in ep.values():
                # agent_data["infos"] is a list of dicts, one per timestep
                vals = [
                    average_over_regions(info.get(metric, np.nan))
                    for info in agent_data["infos"]
                ]
                vals = np.array(vals, dtype=float)
                if not np.all(np.isnan(vals)):
                    all_values.append(vals)

        if not all_values:
            # Skip metrics with no data
            continue

        # Plot
        plt.figure(figsize=(6, 4))
        for vals in all_values:
            plt.plot(vals, alpha=0.4)
        avg_curve = np.nanmean(np.stack(all_values), axis=0)
        plt.plot(avg_curve, color="black", linewidth=2, label="Mean")

        plt.title(label)
        plt.xlabel("Timestep")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(eval_dir / f"{metric}.pdf", dpi=150)
        plt.close()

    print(f"✅ Info variable plots saved under {eval_dir}")


def run_all_feature_plots(episodes, eval_dir: Path, num_agents=None):
    """
    Run all feature evaluation plots and CSVs:
    - Compute basic metrics
    - Plot rewards
    - Plot info variables
    - Plot global variables combined
    - Plot per-region metrics (if num_agents given)
    - Save CSV summaries
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Compute basic metrics and save CSV
    metrics = compute_basic_metrics(episodes)
    metrics_csv = eval_dir / "metrics_summary.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(f"💾 Metrics summary saved to {metrics_csv}")

    # 2️⃣ Plot RL rewards
    try:
        plot_rewards(episodes, eval_dir)
    except Exception as e:
        print(f"⚠️ Reward plot failed: {e}")

    # 3️⃣ Plot info variables
    try:
        plot_info_variables(episodes, eval_dir)
    except Exception as e:
        print(f"⚠️ Info variable plots failed: {e}")

    # 4️⃣ Plot global variables combined
    try:
        plot_global_variables_combined(episodes, eval_dir)
    except Exception as e:
        print(f"⚠️ Global variable plots failed: {e}")

    # 5️⃣ Per-region metric plots
    if num_agents:
        try:
            plot_metrics_per_region(episodes, eval_dir, regions=list(range(num_agents)))
        except Exception as e:
            print(f"⚠️ Per-region metric plots failed: {e}")

    print(f"✅ All feature evaluation plots and CSVs saved → {eval_dir}")
