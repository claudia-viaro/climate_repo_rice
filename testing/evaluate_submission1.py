"""
Evaluation script for the rice environment
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logger

_path = Path(os.path.abspath(__file__))

from my_project.utils.fixed_paths import PUBLIC_REPO_DIR
from my_project.utils.run_unittests import fetch_base_env
from gymnasium.spaces import MultiDiscrete

# climate-cooperation-competition
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
logging.info("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

logging.getLogger().setLevel(logging.ERROR)

_EVAL_SEED = 1234567890  # seed used for evaluation

_INDEXES_FILENAME = "climate_economic_min_max_indices.txt"

_METRICS_TO_LABEL_DICT = OrderedDict()
# Read the dict values below as
# (label, decimal points used to round off value: 0 becomes an integer)

_METRICS_TO_LABEL_DICT["global_temperature"] = ("Temperature Rise", 2)
_METRICS_TO_LABEL_DICT["global_carbon_mass"] = ("Carbon Mass", 2)
_METRICS_TO_LABEL_DICT["capital_all_regions"] = ("Capital", 2)
_METRICS_TO_LABEL_DICT["labor_all_regions"] = ("Labor", 2)
_METRICS_TO_LABEL_DICT["production_factor_all_regions"] = ("Production Factor", 2)
_METRICS_TO_LABEL_DICT["production_all_regions"] = ("Production", 2)
_METRICS_TO_LABEL_DICT["intensity_all_regions"] = ("Intensity", 2)
# _METRICS_TO_LABEL_DICT["global_exegenous_emissions"] = ("Exogenous Emissions", 2)
_METRICS_TO_LABEL_DICT["global_land_emissions"] = ("Land Emissions", 2)
# _METRICS_TO_LABEL_DICT["capital_deprication_all_regions"] = ("Capital Deprication", 2)
_METRICS_TO_LABEL_DICT["savings_all_regions"] = ("Savings", 2)
_METRICS_TO_LABEL_DICT["mitigation_rates_all_regions"] = ("Mitigation Rate", 0)
_METRICS_TO_LABEL_DICT["export_limit_all_regions"] = ("Max Export Limit", 2)
_METRICS_TO_LABEL_DICT["mitigation_cost_all_regions"] = ("Mitigation Cost", 2)
_METRICS_TO_LABEL_DICT["damages_all_regions"] = ("Damages", 2)
_METRICS_TO_LABEL_DICT["abatement_cost_all_regions"] = ("Abatement Cost", 2)
_METRICS_TO_LABEL_DICT["utility_all_regions"] = ("Utility", 2)
_METRICS_TO_LABEL_DICT["social_welfare_all_regions"] = ("Social Welfare", 2)
_METRICS_TO_LABEL_DICT["reward_all_regions"] = ("Reward", 2)
# _METRICS_TO_LABEL_DICT["consumption_all_regions"] = ("Consumption", 2)
_METRICS_TO_LABEL_DICT["current_balance_all_regions"] = ("Current Balance", 2)
_METRICS_TO_LABEL_DICT["gross_output_all_regions"] = ("Gross Output", 2)
_METRICS_TO_LABEL_DICT["investment_all_regions"] = ("Investment", 2)
_METRICS_TO_LABEL_DICT["production_all_regions"] = ("Production", 2)
_METRICS_TO_LABEL_DICT["minimum_mitigation_rate_all_regions"] = (
    "Minimum Mitigation Rate",
    0,
)
# _METRICS_TO_LABEL_DICT["aux_m_all_regions"] = ("Emissions", 2)


# to run is
# python ./scripts/example_2.py --results_dir ./Submissions/1751377309.zip


def get_output_dir(results_dir):
    """Generate output directory name based on results_dir."""
    base_name = os.path.splitext(os.path.basename(results_dir))[
        0
    ]  # Get filename without .zip
    output_dir = os.path.join("outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)  # Create if it doesn't exist
    return output_dir


def get_imports(framework=None):
    """
    Fetch relevant imports.
    """
    assert framework is not None
    from debug_scripts.debug_trainer import (
        create_trainer,
        fetch_episode_states,
        load_model_checkpoints,
        set_num_agents,
    )
    from debug_scripts.debug_trainer import set_num_agents

    return create_trainer, load_model_checkpoints, fetch_episode_states, set_num_agents


def try_to_unzip_file(path):
    """
    Obtain the 'results' directory from the system arguments.
    """
    try:
        _unzipped_dir = os.path.join("/tmp", str(time.time()))
        shutil.unpack_archive(path, _unzipped_dir)
        return _unzipped_dir
    except Exception as err:
        raise ValueError("Cannot obtain the results directory") from err


def validate_dir(results_dir=None):
    """
    Validate that all the required files are present in the 'results' directory.
    """
    assert results_dir is not None
    framework = None

    files = os.listdir(results_dir)

    if ".rllib" in files:
        framework = "rllib"
        # RLlib was used for training
        for file in ["rice.py", "rice_helpers.py"]:
            if file not in files:
                success = False
                logging.error(
                    "%s is not present in the results directory: %s!",
                    file,
                    results_dir,
                )
                comment = f"{file} is not present in the results directory!"
                break
            yaml_success = False
            for file in [
                "rice_rllib_discrete.yaml",
                "rice_rllib_cont.yaml",
                "rice_rllib_cont_beta.yaml",
            ]:
                if file in files:
                    yaml_success = True
                    discrete = "discrete" in file

            if not yaml_success:
                logging.error(
                    "No yaml is present in the results directory: %s!",
                    file,
                    results_dir,
                )
                comment = f"yaml is not present in the results directory!"
                break
            success = True
            comment = "Valid submission"
    else:
        success = False
        logging.error(
            "Missing identifier file! "
            "file must be present in the results directory: %s",
            results_dir,
        )
        comment = "Missing identifier file!"
    print("comment", comment)
    return framework, success, comment, discrete


def compute_metrics(
    fetch_episode_states,
    trainer,
    framework,
    num_episodes=1,
    include_c_e_idx=True,
    log_config=None,
    file_name=None,
):
    """
    Generate episode rollouts and compute metrics.
    """
    assert trainer is not None

    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())
    # Add auxiliary outputs required for processing
    required_outputs = desired_outputs + ["activity_timestep"]
    log_file = log_config["log_file"]
    plot_dir = log_config["plot_dir"]

    # Function to write logs
    def write_log(message):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        logging.info(message)

    log_config["enabled"] = True
    # if log_config and log_config["enabled"]:
    #    wandb_config = log_config["wandb_config"]
    #    wandb.login(key=wandb_config["login"])
    #    wandb.init(project=wandb_config["project"],
    #        name=f'{wandb_config["run"]}_eval',
    #        entity=wandb_config["entity"])

    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            if fetch_episode_states is not None:
                episode_states[episode_id] = fetch_episode_states(
                    trainer, required_outputs, file_name
                )
            else:
                episode_states[episode_id] = trainer.fetch_episode_global_states(
                    required_outputs
                )

        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            elif feature == "gross_output_all_regions":
                for episode_id in range(num_episodes):
                    # collect gross output results based on activity timestep
                    activity_timestep = episode_states[episode_id]["activity_timestep"]
                    activity_index = np.append(
                        1.0, np.diff(activity_timestep.squeeze())
                    )
                    activity_index = [np.isclose(v, 1.0) for v in activity_index]
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id]["gross_output_all_regions"][
                            activity_index
                        ]
                    )

            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )

            if log_config and log_config["enabled"]:
                # TODO: fix dirty method to remove negotiation steps from results
                interval = (len(episode_states[episode_id][feature]) - 1) // 20
                ys = episode_states[episode_id][feature][0::interval].T

                xs = list(range(len(ys[0])))
                plot_name = feature.replace("_", " ").capitalize()
                plt.figure(figsize=(10, 5))  # Create a new figure
                if feature == "global_temperature":
                    plt.plot(xs, ys[0], label="Atmosphere")
                    plt.plot(xs, ys[1], label="Ocean")
                    plt.title(plot_name)
                    plt.xlabel("Step")
                    plt.ylabel("Temperature")
                    plt.legend()
                    # plot = wandb.plot.line_series(
                    #    xs=xs,
                    #    ys=ys.tolist(),
                    #    keys=["Atmosphere", "Ocean"],
                    #    title=plot_name,
                    #    xname="step",
                    # )
                    write_log(f"Plot saved: {plot_name}")
                    # wandb.log({plot_name: plot})
                elif feature == "global_carbon_mass":
                    plt.plot(xs, ys[0], label="Atmosphere")
                    plt.plot(xs, ys[1], label="Upper ocean")
                    plt.plot(xs, ys[2], label="Lower ocean")
                    plt.title(plot_name)
                    plt.xlabel("Step")
                    plt.ylabel("Carbon Mass")
                    plt.legend()
                    # plot = wandb.plot.line_series(
                    #    xs=xs,
                    #    ys=ys.tolist(),
                    #    keys=["Atmosphere", "Upper ocean", "Lower ocean"],
                    #    title=plot_name,
                    #    xname="step",
                    # )
                    # wandb.log({plot_name: plot})
                    write_log(f"Plot saved: {plot_name}")
                elif feature.endswith("_all_regions"):
                    value_name = feature[:-12].replace("_", " ")
                    plot_name = value_name.capitalize()
                    plot_name_mean = f"Mean {value_name}"
                    ys_mean = np.mean(ys, axis=0)
                    data = [[x, y] for (x, y) in zip(xs, ys_mean.tolist())]

                    # Plot mean across regions
                    plt.plot(
                        xs, ys_mean, label=plot_name_mean, color="black", linestyle="--"
                    )

                    # Plot each region separately
                    for i in range(len(ys)):
                        plt.plot(xs, ys[i], label=f"Region {i}")

                    plt.title(plot_name)
                    plt.xlabel("Step")
                    plt.ylabel(value_name.capitalize())
                    plt.legend()
                    """
                    table = wandb.Table(data=data, columns=["step", value_name])
                    plot_mean = wandb.plot.line(
                        table, "step", value_name, title=plot_name_mean
                    )
                    plot = wandb.plot.line_series(
                        xs=xs,
                        ys=ys.tolist(),
                        keys=[f"Region {x}" for x in range(len(ys))],
                        title=plot_name,
                        xname="step",
                    )
                    wandb.log({plot_name_mean: plot_mean})
                    wandb.log({plot_name: plot})
                    """
                    write_log(f"Plot saved: {plot_name_mean}")
                    write_log(f"Plot saved: {plot_name}")
                # Save the plot
                plot_path = os.path.join(plot_dir, f"{feature}.png")
                plt.savefig(plot_path, dpi=300)
                plt.close()
                write_log(f"Plot saved: {plot_path}")

        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    return success, comment, eval_metrics


def val_metrics(logged_ts, framework, num_episodes=1, include_c_e_idx=True):
    """
    Generate episode rollouts and compute metrics.
    """
    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())
    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            episode_states[episode_id] = logged_ts

        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            elif feature == "gross_output_all_regions":
                for episode_id in range(num_episodes):
                    # collect gross output results based on activity timestep
                    activity_timestep = episode_states[episode_id]["activity_timestep"]
                    activity_index = np.append(
                        1.0, np.diff(activity_timestep.squeeze())
                    )
                    activity_index = [np.isclose(v, 1.0) for v in activity_index]
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id]["gross_output_all_regions"][
                            activity_index
                        ]
                    )
            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )
        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    return success, comment, eval_metrics


def perform_format(val, num_decimal_places):
    """
    Format value to the number of desired decimal points.
    """
    if np.isnan(val):
        return val
    assert num_decimal_places >= 0
    rounded_val = np.round(val, num_decimal_places)
    if num_decimal_places == 0:
        return int(rounded_val)
    return rounded_val


def perform_evaluation(
    results_directory, framework, num_episodes=1, discrete=True, eval_seed=None
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None
    assert num_episodes > 0

    (create_trainer, load_model_checkpoints, fetch_episode_states, set_num_agents) = (
        get_imports(framework=framework)
    )

    # Load a run configuration
    yaml_path = f"rice_{framework}_discrete.yaml"
    config_file = os.path.join(results_directory, yaml_path)

    try:
        assert os.path.exists(config_file)
    except Exception as err:
        logging.error(f"The run configuration is missing in {results_directory}.")
        raise err

    with open(config_file, "r", encoding="utf-8") as file_ptr:
        run_config = yaml.safe_load(file_ptr)
        # force eval on single worker
        run_config["trainer"]["num_workers"] = 0
        log_config = run_config["logging"]
    # Replace wandb-specific logging settings
    log_config["output_dir"] = os.path.join(results_directory, "logs")
    log_config["plot_dir"] = os.path.join(results_directory, "plots")

    # Ensure directories exist
    os.makedirs(log_config["output_dir"], exist_ok=True)
    os.makedirs(log_config["plot_dir"], exist_ok=True)

    # Create a log file
    log_file_path = os.path.join(log_config["output_dir"], "evaluation_log.txt")
    log_config["log_file"] = log_file_path
    # update region yamls
    set_num_agents(run_config)

    # Copy the PUBLIC region yamls and rice_build.cu to the results directory.
    if not os.path.exists(os.path.join(results_directory, "region_yamls")):
        shutil.copytree(
            os.path.join(PUBLIC_REPO_DIR, "region_yamls"),
            os.path.join(results_directory, "region_yamls"),
        )
    if not os.path.exists(os.path.join(results_directory, "rice_build.cu")):
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "rice_build.cu"),
            os.path.join(results_directory, "rice_build.cu"),
        )

    # Create Trainer object
    try:
        trainer = create_trainer(
            run_config, source_dir=results_directory, seed=eval_seed
        )

    except Exception as err:
        logging.error(f"Could not create Trainer with the run_config provided.")
        raise err

    # Load model checkpoints
    try:
        load_model_checkpoints(trainer, results_directory)
    except Exception as err:
        logging.error(f"Could not load model checkpoints.")
        raise err

    # Compute metrics
    try:
        success, comment, eval_metrics = compute_metrics(
            fetch_episode_states,
            trainer,
            framework,
            num_episodes=num_episodes,
            log_config=log_config,
            file_name=run_config["env"]["scenario"],
        )

        if framework == "warpdrive":
            trainer.graceful_close()

        return success, eval_metrics, comment

    except Exception as err:
        logging.error(f"Count not fetch episode and compute metrics.")
        raise err


def get_temp_rise_and_gross_output(env, actions):
    env.reset()
    for _ in range(env.episode_length):
        env.step(actions)
    temperature_array = env.global_state["global_temperature"]["value"]
    temperature_rise = temperature_array[-1, 0] - temperature_array[0, 0]

    total_gross_production = np.sum(
        env.global_state["gross_output_all_regions"]["value"]
    )
    return temperature_rise, total_gross_production


def generate_min_max_climate_economic_indices():
    """
    Generate min and max climate and economic indices for the leaderboard.
    0% savings, 100% mitigation => best climate index, worst economic index
    100% savings, 0% mitigation => worst climate index, best economic index
    """
    env = fetch_base_env()  # base rice env
    assert isinstance(
        env.action_space[0], MultiDiscrete
    ), "Unknown action space for env."
    all_zero_actions = {
        agent_id: np.zeros(
            len(env.action_space[agent_id].nvec),
            dtype=np.int32,
        )
        for agent_id in range(env.num_agents)
    }

    # 0% savings, 100% mitigation
    low_savings_high_mitigation_actions = {}
    savings_action_idx = 0
    mitigation_action_idx = 1
    for agent_id in range(env.num_agents):
        low_savings_high_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        low_savings_high_mitigation_actions[agent_id][
            mitigation_action_idx
        ] = env.num_discrete_action_levels
    # Best climate index, worst economic index
    best_ci, worst_ei = get_temp_rise_and_gross_output(
        env, low_savings_high_mitigation_actions
    )

    high_savings_low_mitigation_actions = {}
    for agent_id in range(env.num_agents):
        high_savings_low_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        high_savings_low_mitigation_actions[agent_id][
            savings_action_idx
        ] = env.num_discrete_action_levels
    worst_ci, best_ei = get_temp_rise_and_gross_output(
        env, high_savings_low_mitigation_actions
    )

    index_dict = {
        "min_ci": float(worst_ci),
        "max_ci": float(best_ci),
        "min_ei": float(worst_ei),
        "max_ei": float(best_ei),
    }
    return index_dict


if __name__ == "__main__":
    logger.init_logger("evaluing_file.jsonl")

    logging.info("Starting evaluation script.")

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        required=True,
        help="Path to the submission zip file or directory.",
    )
    args = parser.parse_args()

    # Ensure results_dir exists
    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(f"Error: {args.results_dir} does not exist!")

    # Generate output directory dynamically
    output_dir = get_output_dir(args.results_dir)

    # Configure logging
    log_file = os.path.join(output_dir, "evaluation_log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Using submission files in {args.results_dir}")
    logging.info(f"Saving logs and plots in {output_dir}")

    # Extract if results_dir is a zip file
    if args.results_dir.endswith(".zip"):
        extracted_dir = os.path.join(
            "Submissions", os.path.splitext(os.path.basename(args.results_dir))[0]
        )
        shutil.unpack_archive(args.results_dir, extracted_dir)
        results_dir = extracted_dir
    else:
        results_dir = args.results_dir
    framework, results_dir_is_valid, comment, discrete = validate_dir(results_dir)
    if not results_dir_is_valid:
        raise AssertionError(f"{results_dir} is not a valid submission directory.")

    # Perform evaluation
    succeeded, metrics, comments = perform_evaluation(
        results_dir,
        framework=framework,  # Replace with actual framework
        discrete=True,
        eval_seed=None,
    )

    # Report results
    eval_result_str = (
        f"Evaluation Succeeded: {succeeded}\nMetrics: {metrics}\nComments: {comments}"
    )
    logging.info(eval_result_str)
    print(eval_result_str)
    print(f"Logging complete. Check the output directory: {output_dir}")
