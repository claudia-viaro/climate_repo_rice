from .rice_env import *
import random
import numpy as np
from math import ceil
import logging, warnings

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
_FEATURES = "features"
_ACTION_MASK = "action_mask"


class ConvergenceMitigationSavings(Rice):
    """
    a simplified action space to test convergence.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="ConvergenceMitigationSavings",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def get_mask_index(self, action_type):
        """get start and end index for a particular action"""

        if action_type == "savings":
            return 0, sum(self.savings_possible_actions)
        if action_type == "mitigation_rates":
            return sum(self.savings_possible_actions), sum(
                self.savings_possible_actions + self.mitigation_rate_possible_actions
            )

    def calc_action_window(self, region_id):
        """
        create mask around all actions not adjacent to the previous action.
        """

        base_mask = self.default_agent_action_mask.copy()
        single_actions = ["savings_all_regions", "mitigation_rates_all_regions"]
        for action in single_actions:
            previous_action = self.global_state[action]["value"][
                max(0, self.current_timestep), region_id
            ]
            previous_action_scaled = int(
                previous_action * self.num_discrete_action_levels
            )
            mask_start, mask_end = self.get_mask_index(
                action.replace("_all_regions", "")
            )

            current_mask = base_mask[mask_start:mask_end]
            current_mask[:] = 0
            current_mask[
                max(0, previous_action_scaled - 1) : min(
                    self.num_discrete_action_levels, previous_action_scaled + 2
                )
            ] = 1
            base_mask[mask_start:mask_end] = current_mask

        return base_mask.astype(int)

    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = (
            self.savings_possible_actions + self.mitigation_rate_possible_actions
        )

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions + self.evaluation_possible_actions
            )

        return total_possible_actions

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "export_limit":
            return self.get_state("export_limit_all_regions", timestep=0)

        if action_type == "import_bids":
            return self.get_state("import_bids_all_regions", timestep=0)

        if action_type == "import_tariffs":
            return self.get_state("import_tariffs", timestep=0)

    def get_actions_index(self, action_type):
        if action_type == "savings":
            return 0
        if action_type == "mitigation_rate":
            return len(self.savings_possible_actions)

    def step_climate_and_economy(self, actions=None, actions_dict=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)
        if actions_dict is None:
            actions_dict = {
                "savings_all_regions": self.get_actions("savings", actions),
                "mitigation_rates_all_regions": self.get_actions(
                    "mitigation_rate", actions
                ),
                "export_limit_all_regions": self.get_actions("export_limit", actions),
                "import_bids_all_regions": self.get_actions("import_bids", actions),
                "import_tariffs_all_regions": self.get_actions(
                    "import_tariffs", actions
                ),
            }
        if self.action_space_type == "continuous":
            actions_dict = self.cont_implement_bounds(actions_dict)
        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(
            gross_outputs, gross_imports, net_imports
        )

        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)

        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = self.generate_info(observations, rewards)

        return observations, rewards, terminateds, truncateds, info


class TradeAwarePartners(Rice):
    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="TradeAwarePartners",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
        step_logger=None,
        region_yamls_path=None,
    ):

        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
            step_logger=step_logger,
            region_yamls_path=region_yamls_path,
        )

    def step_evaluate_proposals(self, actions=None, actions_dict_save=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        # Build actions_dict for this stage
        actions_dict = {
            "proposal_decisions": self.get_actions("proposal_decisions", actions)
        }
        actions_dict = self.apply_fixed_actions_to_vector(actions_dict)

        # --- Trade-based acceptance logic ---
        flip_threshold = 0.4
        base_flip_prob = 0.7

        for region_id in range(self.num_regions):
            for partner_id in range(self.num_regions):
                if region_id == partner_id:
                    continue
                decision = actions_dict["proposal_decisions"][region_id, partner_id]

                if decision == 1:
                    # --- Baseline logic from NewPartners ---
                    last_committed = self.global_state["requested_mitigation_rate"][
                        "value"
                    ][self.current_timestep - 1, partner_id]
                    last_promised = self.global_state["promised_mitigation_rate"][
                        "value"
                    ][self.current_timestep - 1, partner_id]
                    rate_gap = abs(last_committed - last_promised)

                    prob_flip = 0.0
                    if rate_gap > flip_threshold:
                        prob_flip += base_flip_prob

                    # --- NEW: Trade modification ---
                    tariff_level = self.global_state["tariffs"]["value"][
                        self.current_timestep - 1, region_id, partner_id
                    ]
                    trade_volume = (
                        self.global_state["gross_imports"]["value"][
                            self.current_timestep - 1, region_id, partner_id
                        ]
                        + self.global_state["gross_imports"]["value"][
                            self.current_timestep - 1, partner_id, region_id
                        ]
                    )

                    if tariff_level > 0.05:  # strong penalty if tariffs imposed
                        prob_flip += 0.8
                    if trade_volume > 100.0:  # reward strong trade ties
                        prob_flip = max(0.0, prob_flip - 0.3)

                    if np.random.rand() < min(1.0, prob_flip):
                        actions_dict["proposal_decisions"][region_id, partner_id] = 0

        # Update global state and return
        self.set_state("proposal_decisions", actions_dict["proposal_decisions"])
        observations = self.get_observations()
        rewards = {r: 0.0 for r in range(self.num_regions)}
        terminateds = {"__all__": 0, **{r: 0 for r in range(self.num_regions)}}
        truncateds = {"__all__": 0, **{r: 0 for r in range(self.num_regions)}}
        info = {"__common__": {"stage": "evaluate", "timestep": self.current_timestep}}

        return observations, rewards, terminateds, truncateds, info


import networkx as nx


class NegoReward(Rice):
    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="NegoReward",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
        actions_masked=None,
        step_logger=None,
        region_yamls_path=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
            actions_masked=actions_masked,
            step_logger=step_logger,
            region_yamls_path=region_yamls_path,
        )

    def detect_largest_clique(self, mutual: np.ndarray) -> int:
        """
        Brute-force search for largest clique in an undirected graph.

        Args:
            mutual (np.ndarray): symmetric 0/1 adjacency matrix

        Returns:
            int: size of the largest clique
        """
        n = mutual.shape[0]
        best = 0

        def is_clique(nodes):
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if mutual[nodes[i], nodes[j]] == 0:
                        return False
            return True

        def dfs(curr, candidates):
            nonlocal best
            if not candidates:
                best = max(best, len(curr))
                return
            while candidates:
                v = candidates.pop()
                new_curr = curr + [v]
                if is_clique(new_curr):
                    dfs(new_curr, [u for u in candidates if mutual[v, u]])
                else:
                    best = max(best, len(curr))

        dfs([], list(range(n)))
        return best

    def calc_rewards(self, utilities, welfloss_multipliers, save_state=True):
        rewards = np.zeros(self.num_regions, dtype=self.float_dtype)

        # Only read from global state if negotiation is on
        if getattr(self, "negotiation_on", False):
            coop_factor = self.get_state(
                "coop_factor_all_regions"
            )  # shape: (num_regions,)
            mitigation_rates = self.get_state("mitigation_rates_all_regions")
            beta = getattr(self, "coop_beta", 1.0)
        else:
            coop_factor = np.zeros(self.num_regions)
            mitigation_rates = np.zeros(self.num_regions)
            beta = 0.0

        for region_id in range(self.num_regions):
            # Base utility
            if not self.relative_reward:
                rewards[region_id] = (
                    utilities[region_id] * welfloss_multipliers[region_id]
                )
            else:
                base_util = utilities[region_id] * welfloss_multipliers[region_id]
                coop_bonus = 1.0 + beta * (
                    coop_factor[region_id] * mitigation_rates[region_id]
                )
                rewards[region_id] = (
                    base_util * coop_bonus
                    - self.baseline_rice.global_state["reward_all_regions"]["value"][
                        self.activity_timestep, region_id
                    ]
                )

            # Save state
            if save_state:
                self.set_state(
                    "reward_all_regions", rewards[region_id], region_id=region_id
                )

        return rewards

    def detect_largest_clique_with_members(self, mutual: np.ndarray):
        """
        Brute-force search for the largest clique in an undirected graph.

        Args:
            mutual (np.ndarray): symmetric 0/1 adjacency matrix of mutual agreements.

        Returns:
            block_size (int): size of the largest clique
            members (list[int]): indices of regions in the largest clique
        """
        n = mutual.shape[0]
        best_size = 0
        best_members = []

        def is_clique(nodes):
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if not mutual[nodes[i], nodes[j]]:
                        return False
            return True

        def dfs(curr, candidates):
            nonlocal best_size, best_members
            if not candidates:
                if len(curr) > best_size:
                    best_size = len(curr)
                    best_members = curr.copy()
                return
            while candidates:
                v = candidates.pop()
                new_curr = curr + [v]
                if is_clique(new_curr):
                    dfs(new_curr, [u for u in candidates if mutual[v, u]])
                else:
                    if len(curr) > best_size:
                        best_size = len(curr)
                        best_members = curr.copy()

        dfs([], list(range(n)))
        return best_size, best_members

    def reset_state(self, key):
        # call parent reset_state first
        super().reset_state(key)

        # only initialize your new variables at timestep 0
        if key == "coop_factor_all_regions":
            self.set_state(key, value=np.zeros(self.num_regions))
        if key == "climate_penalty_all_regions":
            self.set_state(key, value=np.zeros(self.num_regions))
        if key == "avg_agreement_rate":
            self.set_state(key, value=0.0)  # scalar wrapped in 1-element array
        if key == "largest_block_size":
            self.set_state(key, value=0.0)  # scalar

    def reset(self, *, seed=None, options=None):
        # Call parent reset to initialize all default states
        obs, info = super().reset(seed=seed, options=options)

        # Initialize your new variables at timestep 0
        self.reset_state("coop_factor_all_regions")
        self.reset_state("climate_penalty_all_regions")
        self.reset_state("largest_block_size")  # scalar
        self.reset_state("avg_agreement_rate")  # scalar

        # Return the same observation/info dict
        return obs, info

    def compute_coop_climate_modifiers(self, save_state=True):
        """
        Compute per-region cooperation factors and global climate penalty.

        Returns:
            coop_factor_all_regions (np.ndarray): shape (num_regions,)
            climate_penalty_all_regions (np.ndarray): shape (num_regions,)
        """
        if not getattr(self, "negotiation_on", False):
            # Negotiation off → return zeros
            zeros = np.zeros(self.num_regions, dtype=self.float_dtype)
            return zeros, zeros

        # Proposal decisions: shape (num_regions, num_regions)
        A = self.get_state("proposal_decisions")  # shape (num_regions, num_regions)

        # Mutual agreements (symmetric)
        mutual = (A > 0) & (A.T > 0)

        # Largest clique using original DFS function
        block_size, clique_members = self.detect_largest_clique_with_members(mutual)

        # Per-region cooperation factor: fraction of successful mutual agreements inside the largest clique
        coop_factor_all_regions = np.zeros(self.num_regions, dtype=self.float_dtype)
        for i in range(self.num_regions):
            if i in clique_members:
                # Count mutual agreements with other clique members
                coop_factor_all_regions[i] = np.sum(mutual[i, clique_members]) / max(
                    len(clique_members) - 1, 1
                )
            else:
                coop_factor_all_regions[i] = 0.0

        # Compute average mitigation rate among mutual agreements
        mitigation_rates = self.get_state(
            "mitigation_rates_all_regions"
        )  # shape (num_regions,)
        mutual_rates = [
            mitigation_rates[i]
            for i in range(self.num_regions)
            for j in range(self.num_regions)
            if i != j and mutual[i, j]
        ]
        avg_agreement_rate = float(np.mean(mutual_rates)) if mutual_rates else 0.0

        # Global climate penalty (scalar, repeated per region)
        temp_anomaly = float(self.get_state("global_temperature")[0])
        climate_penalty_all_regions = np.full(
            self.num_regions, -temp_anomaly, dtype=self.float_dtype
        )

        # Save state
        if save_state:
            self.set_state("coop_factor_all_regions", coop_factor_all_regions)
            self.set_state("climate_penalty_all_regions", climate_penalty_all_regions)
            self.set_state("largest_block_size", float(block_size))
            self.set_state("avg_agreement_rate", avg_agreement_rate)

        return coop_factor_all_regions, climate_penalty_all_regions

    def step_climate_and_economy(self, actions=None, actions_dict=None):
        # 1. Compute timestep and validate
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)

        # 2. Build actions_dict if not provided
        if actions_dict is None:
            actions_dict = {
                "savings_all_regions": self.get_actions("savings", actions),
                "mitigation_rates_all_regions": self.get_actions(
                    "mitigation_rate", actions
                ),
                "export_limit_all_regions": self.get_actions("export_limit", actions),
                "import_bids_all_regions": self.get_actions("import_bids", actions),
                "import_tariffs_all_regions": self.get_actions(
                    "import_tariffs", actions
                ),
            }

        # 3. Apply fixed actions
        actions_dict = self.apply_fixed_actions_to_vector(actions_dict)

        # 4. Set actions into global state
        self.set_actions_in_global_state(actions_dict)

        # 5. Compute core climate & economy variables
        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()
        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )
        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )
        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)
        self.calc_social_welfares(utilities)
        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)
        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        # 6. Compute the new negotiation / cooperation features **before observations**
        coop_factor, climate_penalty = self.compute_coop_climate_modifiers(
            save_state=True
        )

        current_simulation_year = self.calc_current_simulation_year()

        # 7. Get observations (now includes the new features)
        observations = self.get_observations()

        # 8. Compute rewards (reads coop/climate modifiers from global_state)
        rewards = self.get_rewards()

        # 9. Debug print
        # 10. Termination / truncation
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {"__all__": current_simulation_year == self.episode_length}

        if self.is_baseline:
            # Minimal info for debug
            info = {
                "__common__": {
                    "stage": "climate",
                    "timestep": self.current_timestep,
                    "rl": False,
                }
            }
        else:
            if terminateds["__all__"]:
                info_rl = self._get_rl_episode_info()
                info_diffs = self._get_climate_episode_info()
                self.last_info_rl = info_rl
                self.last_info_diffs = info_diffs

            info = {
                "__common__": {
                    "stage": "climate",
                    "timestep": self.current_timestep,
                    "rl": True,
                }
            }

        return observations, rewards, terminateds, truncateds, info

    def _get_climate_episode_info(self):
        """
        Collect per-timestep climate info (global + per-region diffs)
        for the entire episode, relative to timestep 0.
        Adds mean largest_block_size across timesteps at the end.
        """

        episode_length = self.episode_length + 1
        climate_history = []

        # Reference values at timestep 0
        temp0_rl = self.get_state("global_temperature", timestep=0)[0]
        temp0_bau = self.baseline_rice.get_state("global_temperature", timestep=0)[0]

        block_sizes = []

        for t in range(episode_length):
            if self.global_state["negotiation_stage"]["value"][t] != 0:
                continue

            # Align BAU using activity timestep
            act_t = self.global_state["activity_timestep"]["value"][t]

            step_info = {"timestep": t, "activity_timestep": act_t}

            # Global temperature rise
            temp_rl = self.get_state("global_temperature", timestep=t)[0]
            temp_bau = self.baseline_rice.get_state(
                "global_temperature", timestep=act_t
            )[0]

            step_info["temp_rise_rl"] = temp_rl - temp0_rl
            step_info["temp_rise_bau"] = temp_bau - temp0_bau
            step_info["temp_rise_bau_rl"] = (
                step_info["temp_rise_rl"] - step_info["temp_rise_bau"]
            )

            # Per-region differences
            # Track largest_block_size if negotiation is on
            if getattr(self, "negotiation_on", False):
                block_size = float(self.get_state("largest_block_size", timestep=t))
                block_sizes.append(block_size)
                step_info["largest_block_size"] = block_size

            climate_history.append(step_info)

        # Append mean largest_block_size as a final entry
        climate_history.append(
            {"largest_block_mean": float(np.mean(block_sizes)) if block_sizes else 0.0}
        )

        return climate_history

    def _get_rl_episode_info(self):

        # --- Accumulate RL features per timestep ---

        rl_features = [
            "global_temperature",
            "global_emissions",
            "capital_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            "savings_all_regions",
            "mitigation_rates_all_regions",
            "current_balance_all_regions",
            # "production_factor_all_regions", exognous
            # "intensity_all_regions", exogenous
            # "mitigation_cost_all_regions", exogenous
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
            "minimum_mitigation_rate_all_regions",
            "promised_mitigation_rate",
            "requested_mitigation_rate",
            "proposal_decisions",
            # "global_cumulative_emissions", # from another carbon mass model, not the base
            "aux_m_all_regions",
            "global_emissions",
        ]

        episode_length = self.episode_length  # include initial timestep
        rl_history = []

        for t in range(episode_length + 1):
            activity = self.global_state["activity_timestep"]["value"][t]
            negotiation = self.global_state["negotiation_stage"]["value"][t]

            # Keep only actual climate steps
            if negotiation != 0 or activity == 0:
                continue

            step_rl = {"timestep": activity}
            for var in rl_features:
                val = self.global_state[var]["value"][t]
                step_rl[var] = val.tolist() if not np.isscalar(val) else val
            rl_history.append(step_rl)

        # ✅ Debug print: show structure without flooding

        return rl_history


class AsymmetricPartners(Rice):
    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="AsymmetricPartners",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
        step_logger=None,
        region_yamls_path=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
            step_logger=step_logger,
            region_yamls_path=region_yamls_path,
        )

        def step_evaluate_proposals(self, actions=None, actions_dict_save=None):
            self.is_valid_negotiation_stage(negotiation_stage=2)
            self.is_valid_actions_dict(actions)

            actions_dict = {
                "proposal_decisions": self.get_actions("proposal_decisions", actions)
            }
            actions_dict = self.apply_fixed_actions_to_vector(actions_dict)

            # --- Non-binding / power asymmetry logic ---
            for region_id in range(self.num_regions):
                for partner_id in range(self.num_regions):
                    if region_id == partner_id:
                        continue
                    decision = actions_dict["proposal_decisions"][region_id, partner_id]

                    if decision == 1:
                        gdp_region = self.global_state["gross_output_all_regions"][
                            "value"
                        ][self.current_timestep - 1, region_id]
                        gdp_partner = self.global_state["gross_output_all_regions"][
                            "value"
                        ][self.current_timestep - 1, partner_id]
                        pop_region = self.global_state["labor_all_regions"]["value"][
                            self.current_timestep - 1, region_id
                        ]
                        pop_partner = self.global_state["labor_all_regions"]["value"][
                            self.current_timestep - 1, partner_id
                        ]

                        gdp_pc_region = gdp_region / max(1, pop_region)
                        gdp_pc_partner = gdp_partner / max(1, pop_partner)

                        power_ratio = gdp_pc_region / max(1e-6, gdp_pc_partner)

                        # If region is "much bigger", agreements may not be binding
                        if power_ratio > 2.0:
                            nonbinding_prob = min(1.0, 0.2 + 0.1 * (power_ratio - 2))
                            if np.random.rand() < nonbinding_prob:
                                actions_dict["proposal_decisions"][
                                    region_id, partner_id
                                ] = 0

            # Update global state and return
            self.set_state("proposal_decisions", actions_dict["proposal_decisions"])
            observations = self.get_observations()
            rewards = {r: 0.0 for r in range(self.num_regions)}
            terminateds = {"__all__": 0, **{r: 0 for r in range(self.num_regions)}}
            truncateds = {"__all__": 0, **{r: 0 for r in range(self.num_regions)}}
            info = {
                "__common__": {"stage": "evaluate", "timestep": self.current_timestep}
            }

            return observations, rewards, terminateds, truncateds, info


class NewPartners(Rice):
    """ """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="NewPartners",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def step_evaluate_proposals(self, actions=None, actions_dict_save=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        # Build actions_dict for this stage
        actions_dict = {
            "proposal_decisions": self.get_actions("proposal_decisions", actions)
        }

        # Apply fixed actions (from YAML)
        actions_dict = self.apply_fixed_actions_to_vector(actions_dict)

        # --- NEW: Probabilistic flip logic ---
        flip_threshold = 0.4  # rate difference threshold
        base_flip_prob = 0.7  # base probability to flip if condition met

        for region_id in range(self.num_regions):
            for partner_id in range(self.num_regions):
                if region_id == partner_id:
                    continue  # skip self
                current_decision = actions_dict["proposal_decisions"][
                    region_id, partner_id
                ]

                # Only consider previously accepted proposals
                if current_decision == 1:
                    # Prior step committed & promised rates
                    last_committed = self.global_state["requested_mitigation_rate"][
                        "value"
                    ][self.current_timestep - 1, partner_id]
                    last_promised = self.global_state["promised_mitigation_rate"][
                        "value"
                    ][self.current_timestep - 1, partner_id]
                    rate_gap = abs(last_committed - last_promised)

                    # Check if partner is “new”
                    prior_decision = self.global_state["proposal_decisions"]["value"][
                        self.current_timestep - 1, partner_id
                    ]
                    new_partner = prior_decision == 0

                    # Compute flip probability
                    prob_flip = 0.0
                    if rate_gap > flip_threshold:
                        prob_flip += base_flip_prob
                    if new_partner:
                        prob_flip += (
                            base_flip_prob * 0.5
                        )  # smaller boost for new partners
                    prob_flip = min(1.0, prob_flip)

                    # Flip probabilistically
                    if np.random.rand() < prob_flip:
                        actions_dict["proposal_decisions"][region_id, partner_id] = 0

        # --- End flip logic ---

        # Update global state
        self.set_state("proposal_decisions", actions_dict["proposal_decisions"])

        for region_id in range(self.num_regions):
            min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)
            self.set_state(
                "minimum_mitigation_rate_all_regions",
                min_mitigation,
                timestep=self.current_timestep,
                region_id=region_id,
            )

        # Observations and rewards
        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {
            "__common__": {
                "stage": "evaluate",
                "timestep": self.current_timestep,
            }
        }
        return observations, rewards, terminateds, truncateds, info

    def step_climate_and_economy(self, actions=None, actions_dict=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)
        if actions_dict is None:
            actions_dict = {
                "savings_all_regions": self.get_actions("savings", actions),
                "mitigation_rates_all_regions": self.get_actions(
                    "mitigation_rate", actions
                ),
                "export_limit_all_regions": self.get_actions("export_limit", actions),
                "import_bids_all_regions": self.get_actions("import_bids", actions),
                "import_tariffs_all_regions": self.get_actions(
                    "import_tariffs", actions
                ),
            }
        if self.action_space_type == "continuous":
            actions_dict = self.cont_implement_bounds(actions_dict)
        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(
            gross_outputs, gross_imports, net_imports
        )

        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)

        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = self.generate_info(observations, rewards)

        return observations, rewards, terminateds, truncateds, info

    def proposal_similarity(self, region_id, partner_id):
        """
        Returns similarity [0,1] between previous committed rates and
        current proposed rates.
        """
        # last committed rate (previous timestep)
        last_rate = self.global_state["requested_mitigation_rate"]["value"][
            self.current_timestep - 1, partner_id
        ]
        # current proposed rate
        current_rate = self.global_state["promised_mitigation_rate"]["value"][
            self.current_timestep - 1, partner_id
        ]
        diff = abs(last_rate - current_rate)
        similarity = max(0, 1 - diff / 0.9)  # normalized to [0,1]
        return similarity


class Convergence(Rice):
    """
    a simplified action space to test convergence.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="Convergence",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def get_mask_index(self, action_type):
        """get start and end index for a particular action"""

        if action_type == "mitigation_rates":
            return 0, sum(self.mitigation_rate_possible_actions)

    def calc_action_window(self, region_id):
        """
        create mask around all actions not adjacent to the previous action.
        """

        base_mask = self.default_agent_action_mask.copy()
        single_actions = [
            "mitigation_rates_all_regions",
        ]
        for action in single_actions:
            previous_action = self.global_state[action]["value"][
                max(0, self.current_timestep), region_id
            ]
            previous_action_scaled = int(
                previous_action * self.num_discrete_action_levels
            )
            mask_start, mask_end = self.get_mask_index(
                action.replace("_all_regions", "")
            )

            current_mask = base_mask[mask_start:mask_end]
            current_mask[:] = 0
            current_mask[
                max(0, previous_action_scaled - 1) : min(
                    self.num_discrete_action_levels, previous_action_scaled + 2
                )
            ] = 1
            base_mask[mask_start:mask_end] = current_mask

        return base_mask.astype(int)

    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = self.mitigation_rate_possible_actions

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions + self.evaluation_possible_actions
            )

        return total_possible_actions

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            return self.get_state("savings_all_regions", timestep=0)

        if action_type == "export_limit":
            return self.get_state("export_limit_all_regions", timestep=0)

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_bids":
            return self.get_state("import_bids_all_regions", timestep=0)

        if action_type == "import_tariffs":
            return self.get_state("import_tariffs", timestep=0)

    def get_actions_index(self, action_type):
        if action_type == "mitigation_rate":
            return 0

    def step_climate_and_economy(self, actions=None, actions_dict=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)
        if actions_dict is None:
            actions_dict = {
                "savings_all_regions": self.get_actions("savings", actions),
                "mitigation_rates_all_regions": self.get_actions(
                    "mitigation_rate", actions
                ),
                "export_limit_all_regions": self.get_actions("export_limit", actions),
                "import_bids_all_regions": self.get_actions("import_bids", actions),
                "import_tariffs_all_regions": self.get_actions(
                    "import_tariffs", actions
                ),
            }
        if self.action_space_type == "continuous":
            actions_dict = self.cont_implement_bounds(actions_dict)
        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(
            gross_outputs, gross_imports, net_imports
        )

        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)

        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = self.generate_info(observations, rewards)

        return observations, rewards, terminateds, truncateds, info


class BasicClubFixed(Rice):
    """
    Club members have a fixed rate and fixed initial members
    non-members can either accept or reject a proposal from the xlub
    -
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="BasicClubTariffAmbition",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )
        self.club_level = 5

    def calc_possible_actions(self, action_type):
        if self.action_space_type == "discrete":
            if action_type == "savings":
                return [self.num_discrete_action_levels]
            if action_type == "mitigation_rate":
                return [self.num_discrete_action_levels]
            if action_type == "export_limit":
                return [self.num_discrete_action_levels]
            if action_type == "import_bids":
                return [self.num_discrete_action_levels] * self.num_regions
            if action_type == "import_tariffs":
                return [self.num_discrete_action_levels] * self.num_regions

            if action_type == "proposal":
                return [self.num_discrete_action_levels]

            if action_type == "proposal_decisions":
                return [2] * self.num_regions

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "export_limit":
            export_action_index = self.get_actions_index("export_limit")
            return [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_bids":
            tariffs_action_index = self.get_actions_index("import_bids")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposed_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")

            return [
                actions[region_id][proposal_actions_index_start]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposal_decisions":
            proposal_decisions_index_start = self.get_actions_index(
                "proposal_decisions"
            )
            num_evaluation_actions = len(self.evaluation_possible_actions)

            proposal_decisions = np.array(
                [
                    actions[region_id][
                        proposal_decisions_index_start : proposal_decisions_index_start
                        + num_evaluation_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )
            for region_id in range(self.num_regions):
                proposal_decisions[region_id, region_id] = 0

            return proposal_decisions

    def step_propose(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        proposed_mitigation_rates = self.get_actions(
            "proposed_mitigation_rate", actions
        )
        self.set_state("proposed_mitigation_rate", np.array(proposed_mitigation_rates))

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info

    def reset_state(self, key):

        if key == "proposed_mitigation_rate":
            self.set_state(key, value=np.zeros(self.num_regions))
        else:
            super().reset_state(key)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # scenario specific global state
        self.reset_state("proposed_mitigation_rate")

        return obs, info

    def calc_mitigation_rate_lower_bound(self, region_id):

        # get all proposed_mitigation rates
        current_proposals = self.global_state["proposed_mitigation_rate"]["value"][
            self.current_timestep
        ]
        proposal_decisions = [
            self.global_state["proposal_decisions"]["value"][
                self.current_timestep, j, region_id
            ]
            for j in range(self.num_regions)
        ]

        # remove all rejected mitigation rates
        accepted_proposals = current_proposals * proposal_decisions

        # return max of accepted
        return max(accepted_proposals)

    def step_evaluate_proposals(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        proposal_decisions = self.get_actions("proposal_decisions", actions)

        self.set_state("proposal_decisions", proposal_decisions)

        for region_id in range(self.num_regions):
            min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)

            self.set_state(
                "minimum_mitigation_rate_all_regions", min_mitigation, region_id
            )

        observations = self.get_observations()

        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}
        return observations, rewards, terminateds, truncateds, info

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            # only applies to original club_members
            if region_id in self.club_members:

                # fix propose a constant rate
                proposal_mask_start, proposal_mask_end = self.get_mask_index("proposal")
                proposal_mask = (
                    [0] * (self.club_level)
                    + [1]
                    + [0] * (self.num_discrete_action_levels - self.club_level - 1)
                )
                mask[proposal_mask_start:proposal_mask_end] = proposal_mask

            # non-members cannot make proposals (ie they propose 0)
            else:
                proposal_mask_start, proposal_mask_end = self.get_mask_index("proposal")
                proposal_mask = [1] + [0] * (self.num_discrete_action_levels - 1)
                mask[proposal_mask_start:proposal_mask_end] = proposal_mask

            # minimum commitment
            min_mitigation_rate = int(
                round(
                    self.get_state(
                        "minimum_mitigation_rate_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            current_mitigation_rate = int(
                round(
                    self.get_state(
                        "mitigation_rates_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            print(region_id, min_mitigation_rate)

            # original members + new joiners
            if region_id in self.club_members or min_mitigation_rate > 0:

                # if agent has a minimum mitigation rate, it must increase mitigation until target reached
                if current_mitigation_rate < min_mitigation_rate:
                    mitigation_mask = (
                        [0] * (current_mitigation_rate + 1)
                        + [1]
                        + [0]
                        * (
                            self.num_discrete_action_levels
                            - current_mitigation_rate
                            - 2
                        )
                    )
                # if at the club level, agent has the possibility of keeping the same mitigation level
                elif (
                    current_mitigation_rate == min_mitigation_rate
                    and current_mitigation_rate < self.num_discrete_action_levels - 1
                ):
                    mitigation_mask = (
                        [0] * (current_mitigation_rate)
                        + [1, 1]
                        + [0]
                        * (
                            self.num_discrete_action_levels
                            - current_mitigation_rate
                            - 2
                        )
                    )
                # if at max mitigation remain there
                elif current_mitigation_rate == self.num_discrete_action_levels - 1:
                    mitigation_mask = [0] * (current_mitigation_rate) + [1]

                # if above club level, normal action window applies
                if current_mitigation_rate > min_mitigation_rate:
                    pass
                else:
                    mitigation_mask_start = sum(self.savings_possible_actions)
                    mitigation_mask_end = mitigation_mask_start + sum(
                        self.mitigation_rate_possible_actions
                    )
                    mask[mitigation_mask_start:mitigation_mask_end] = np.array(
                        mitigation_mask
                    )
            # non club members have no change to the mask
            else:
                pass

            # if in the club (minimum mitigation rate > 0), then tariff
            if min_mitigation_rate > 0:
                print(f"IN CLUB {region_id} {min_mitigation_rate}")
                # tariff non club members
                tariff_mask = []
                for other_region_id in range(self.num_regions):

                    # get other regions mitigation commitment
                    other_mitigation_rate = self.get_state(
                        "minimum_mitigation_rate_all_regions",
                        region_id=other_region_id,
                        timestep=self.current_timestep,
                    )

                    # if other region is self or in club
                    if (other_region_id == region_id) or (
                        other_mitigation_rate >= min_mitigation_rate
                    ):
                        # minimize tariff for free trade
                        regional_tariff_mask = [1] + [0] * (
                            self.num_discrete_action_levels - 1
                        )
                    else:
                        # min tariff by difference between mitigation rate and club mitigation rate
                        tariff_rate = int(min_mitigation_rate - other_mitigation_rate)
                        regional_tariff_mask = [0] * tariff_rate + [1] * (
                            self.num_discrete_action_levels - tariff_rate
                        )
                    tariff_mask.extend(regional_tariff_mask)

                # mask tariff
                tariff_mask_start, tariff_mask_end = self.get_mask_index(
                    "import_tariffs"
                )
                mask[tariff_mask_start:tariff_mask_end] = np.array(tariff_mask)
            # no
            else:
                print(f"NOT IN CLUB:{region_id}")
                pass

            mask_dict[region_id] = mask

        return mask_dict


class MinimalMitigationActionWindow(Rice):
    """Scenario where all agents mitigate to a given extent

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - maximum_mitigation_rate: the rate rate all agents will mitigate to.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="MinimalMitigationActionWindow",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            mitigation_mask = [1] + (self.num_discrete_action_levels - 1) * [0]
            mitigation_mask_start = sum(self.savings_possible_actions)
            mitigation_mask_end = mitigation_mask_start + sum(
                self.mitigation_rate_possible_actions
            )
            mask[mitigation_mask_start:mitigation_mask_end] = np.array(mitigation_mask)

            mask_dict[region_id] = mask

        return mask_dict


class OptimalMitigationActionWindow(Rice):
    """Scenario where all agents mitigate to a given extent

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - maximum_mitigation_rate: the rate rate all agents will mitigate to.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="OptimalMitigationActionWindow",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            current_mitigation_rate = int(
                round(
                    self.get_state(
                        "mitigation_rates_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            # if agent has a minimum mitigation rate, it must increase mitigation until target reached
            if current_mitigation_rate < self.num_discrete_action_levels - 1:
                mitigation_mask = (
                    [0] * (current_mitigation_rate + 1)
                    + [1]
                    + [0]
                    * (self.num_discrete_action_levels - current_mitigation_rate - 2)
                )
            # if at the club level, agent has the possibility of keeping the same mitigation level
            elif current_mitigation_rate == self.num_discrete_action_levels - 1:
                mitigation_mask = [0] * (current_mitigation_rate) + [1]

            mitigation_mask_start = sum(self.savings_possible_actions)
            mitigation_mask_end = mitigation_mask_start + sum(
                self.mitigation_rate_possible_actions
            )
            mask[mitigation_mask_start:mitigation_mask_end] = np.array(mitigation_mask)

            mask_dict[region_id] = mask

        return mask_dict


class BasicClubTariffAmbition(Rice):
    """
    Basic Club adapted to the action window context
     given the action windows all agents start at 0 mitigation and will take some time before they get to high mitigation
     So the tariff is based on the target mitigation rate, ie the agents ambition
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="BasicClubTariffAmbition",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def calc_possible_actions(self, action_type):
        if self.action_space_type == "discrete":
            if action_type == "savings":
                return [self.num_discrete_action_levels]
            if action_type == "mitigation_rate":
                return [self.num_discrete_action_levels]
            if action_type == "export_limit":
                return [self.num_discrete_action_levels]
            if action_type == "import_bids":
                return [self.num_discrete_action_levels] * self.num_regions
            if action_type == "import_tariffs":
                return [self.num_discrete_action_levels] * self.num_regions

            if action_type == "proposal":
                return [self.num_discrete_action_levels]

            if action_type == "proposal_decisions":
                return [2] * self.num_regions

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "export_limit":
            export_action_index = self.get_actions_index("export_limit")
            return [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_bids":
            tariffs_action_index = self.get_actions_index("import_bids")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposed_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")

            return [
                actions[region_id][proposal_actions_index_start]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposal_decisions":
            proposal_decisions_index_start = self.get_actions_index(
                "proposal_decisions"
            )
            num_evaluation_actions = len(self.evaluation_possible_actions)

            proposal_decisions = np.array(
                [
                    actions[region_id][
                        proposal_decisions_index_start : proposal_decisions_index_start
                        + num_evaluation_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )
            for region_id in range(self.num_regions):
                proposal_decisions[region_id, region_id] = 0

            return proposal_decisions

    def step_propose(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        proposed_mitigation_rates = self.get_actions(
            "proposed_mitigation_rate", actions
        )
        self.set_state("proposed_mitigation_rate", np.array(proposed_mitigation_rates))

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info

    def reset_state(self, key):

        if key == "proposed_mitigation_rate":
            self.set_state(key, value=np.zeros(self.num_regions))
        else:
            super().reset_state(key)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # scenario specific global state
        self.reset_state("proposed_mitigation_rate")

        return obs, info

    def calc_mitigation_rate_lower_bound(self, region_id):

        # get all proposed_mitigation rates
        current_proposals = self.global_state["proposed_mitigation_rate"]["value"][
            self.current_timestep
        ]
        proposal_decisions = [
            self.global_state["proposal_decisions"]["value"][
                self.current_timestep, j, region_id
            ]
            for j in range(self.num_regions)
        ]

        # remove all rejected mitigation rates
        accepted_proposals = current_proposals * proposal_decisions

        # return max of accepted
        return max(accepted_proposals)

    def step_evaluate_proposals(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        proposal_decisions = self.get_actions("proposal_decisions", actions)

        self.set_state("proposal_decisions", proposal_decisions)

        for region_id in range(self.num_regions):
            min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)

            self.set_state(
                "minimum_mitigation_rate_all_regions", min_mitigation, region_id
            )

        observations = self.get_observations()

        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}
        return observations, rewards, terminateds, truncateds, info

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            # minimum commitment
            min_mitigation_rate = int(
                round(
                    self.get_state(
                        "minimum_mitigation_rate_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            current_mitigation_rate = int(
                round(
                    self.get_state(
                        "mitigation_rates_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            # if agent has a minimum mitigation rate, it must increase mitigation until target reached
            if current_mitigation_rate < min_mitigation_rate:
                mitigation_mask = (
                    [0] * (current_mitigation_rate + 1)
                    + [1]
                    + [0]
                    * (self.num_discrete_action_levels - current_mitigation_rate - 2)
                )
            # if at the club level, agent has the possibility of keeping the same mitigation level
            elif (
                current_mitigation_rate == min_mitigation_rate
                and current_mitigation_rate < self.num_discrete_action_levels - 1
            ):
                mitigation_mask = (
                    [0] * (current_mitigation_rate)
                    + [1, 1]
                    + [0]
                    * (self.num_discrete_action_levels - current_mitigation_rate - 2)
                )
            # if at max mitigation remain there
            elif current_mitigation_rate == self.num_discrete_action_levels - 1:
                mitigation_mask = [0] * (current_mitigation_rate) + [1]

            # if above club level, normal action window applies
            if current_mitigation_rate > min_mitigation_rate:
                pass
            else:
                mitigation_mask_start = sum(self.savings_possible_actions)
                mitigation_mask_end = mitigation_mask_start + sum(
                    self.mitigation_rate_possible_actions
                )
                mask[mitigation_mask_start:mitigation_mask_end] = np.array(
                    mitigation_mask
                )

            # tariff non club members
            tariff_mask = []
            for other_region_id in range(self.num_regions):

                # get other regions mitigation commitment
                other_mitigation_rate = int(
                    round(
                        self.get_state(
                            "minimum_mitigation_rate_all_regions",
                            region_id=other_region_id,
                            timestep=self.current_timestep,
                        )
                    )
                )

                # if other region is self or in club
                if (other_region_id == region_id) or (
                    other_mitigation_rate >= min_mitigation_rate
                ):
                    # minimize tariff for free trade
                    regional_tariff_mask = [1] + [0] * (
                        self.num_discrete_action_levels - 1
                    )
                else:

                    # min tariff by difference between mitigation rate and club mitigation rate
                    tariff_rate = int(min_mitigation_rate - other_mitigation_rate)
                    regional_tariff_mask = [0] * tariff_rate + [1] * (
                        self.num_discrete_action_levels - tariff_rate
                    )
                tariff_mask.extend(regional_tariff_mask)

            # mask tariff
            tariff_mask_start = sum(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
            )

            tariff_mask_end = tariff_mask_start + sum(
                self.calc_possible_actions("import_tariffs")
            )
            mask[tariff_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask

        return mask_dict


class BasicClubAblateMasks(BasicClubTariffAmbition):
    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="BasicClubAblateMasks",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
        relative_reward=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
            relative_reward=relative_reward,
        )

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            # minimum commitment
            min_mitigation_rate = int(
                round(
                    self.get_state(
                        "minimum_mitigation_rate_all_regions",
                        region_id=region_id,
                        timestep=self.current_timestep,
                    )
                    * self.num_discrete_action_levels
                )
            )

            if self.current_timestep != 0:
                # tariff non club members
                tariff_mask = []
                for other_region_id in range(self.num_regions):

                    # get other regions mitigation commitment
                    other_minimum_mitigation_rate = self.get_state(
                        "minimum_mitigation_rate_all_regions",
                        region_id=other_region_id,
                        timestep=self.current_timestep,
                    )

                    other_mitigation_rate = int(
                        round(
                            self.get_state(
                                "mitigation_rates_all_regions",
                                region_id=other_region_id,
                                timestep=self.current_timestep,
                            )
                            * self.num_discrete_action_levels
                        )
                    )

                    other_previous_mitigation_rate = int(
                        round(
                            self.get_state(
                                "mitigation_rates_all_regions",
                                region_id=other_region_id,
                                timestep=self.current_timestep - 1,
                            )
                            * self.num_discrete_action_levels
                        )
                    )

                    # has other region increased mitigation since previous time step or other region mitigation matches my goal
                    # if yes, no tariff
                    if (
                        other_minimum_mitigation_rate > other_previous_mitigation_rate
                        or other_mitigation_rate >= min_mitigation_rate
                        or other_region_id == region_id
                    ):
                        regional_tariff_mask = [1] + [0] * (
                            self.num_discrete_action_levels - 1
                        )
                    # otherwise tariff on the difference between ambition (minimum mitigation rate) and other regions current mitigation rate
                    else:
                        tariff_rate = int(min_mitigation_rate - other_mitigation_rate)
                        regional_tariff_mask = [0] * tariff_rate + [1] * (
                            self.num_discrete_action_levels - tariff_rate
                        )
                    tariff_mask.extend(regional_tariff_mask)

                # mask tariff
                tariff_mask_start = sum(
                    self.savings_possible_actions
                    + self.mitigation_rate_possible_actions
                    + self.export_limit_possible_actions
                    + self.import_bids_possible_actions
                )

                tariff_mask_end = tariff_mask_start + sum(
                    self.calc_possible_actions("import_tariffs")
                )
                mask[tariff_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask

        return mask_dict


class CarbonLeakageFixed(Rice):
    """
    Scenario to test whether carbon leakage occurs.

    Carbon leakage is an increase of emissions in one region as a
    result of a policy to decrease emissions in another region

    We can check for carbon leakage at the policy level (do they change their mitigation rate)
    and at the emissions level (do their absolute emissions increase)

    Followup experiment
    - create a random club of a given minimum mitigation rate
    - run the rollout with the club and measure emissions of non-club members and measure mitigation rates of non-club members
    - reset the env and re-run the env without the club (self.control = True) and measure the same
    - compare the emissions / mitigation rates of the non-club members in the presence and absence of the club
    - NOTE: it may be that emissions need to be normalized w.r.t. emissions as carbon
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="CarbonLeakage",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
        pct_reward=False,
        clubs_enabled=False,
        club_members=[],
        action_window=True,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
            pct_reward=pct_reward,
            clubs_enabled=clubs_enabled,
            club_members=club_members,
            action_window=action_window,
        )

        self.minimum_mitigation_rate = 8

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            if region_id in self.club_members:
                mitigation_mask = np.array(
                    [0 for _ in range(self.minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels
                            - self.minimum_mitigation_rate
                        )
                    ]
                )

                mask_start = sum(self.savings_possible_actions)
                mask_end = mask_start + sum(self.mitigation_rate_possible_actions)
                mask[mask_start:mask_end] = mitigation_mask
            else:
                pass

            mask_dict[region_id] = mask

        return mask_dict


class CarbonLeakage(Rice):
    """
    Scenario to test whether carbon leakage occurs.

    Carbon leakage is an increase of emissions in one region as a
    result of a policy to decrease emissions in another region

    We can check for carbon leakage at the policy level (do they change their mitigation rate)
    and at the emissions level (do their absolute emissions increase)

    Followup experiment
    - create a random club of a given minimum mitigation rate
    - run the rollout with the club and measure emissions of non-club members and measure mitigation rates of non-club members
    - reset the env and re-run the env without the club (self.control = True) and measure the same
    - compare the emissions / mitigation rates of the non-club members in the presence and absence of the club
    - NOTE: it may be that emissions need to be normalized w.r.t. emissions as carbon
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="CarbonLeakage",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )

        # if its the control group, don't apply the club rules

        self.control = False
        self.training = True
        self.minimum_mitigation_rate = 8
        self.club_size = ceil(self.num_regions / 2)
        self.club_members = random.sample(
            range(0, self.num_regions + 1), self.club_size
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # recreate club each time
        if self.training:
            self.club_members = random.sample(
                range(0, self.num_regions + 1), self.club_size
            )

        # during training, switch up control conditions
        if self.training:
            if random.uniform(0, 1) < 0.3:
                self.control = True
            else:
                self.control = False
        return obs, info

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()

            if region_id in self.club_members and not self.control:

                mitigation_mask = np.array(
                    [0 for _ in range(self.minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels
                            - self.minimum_mitigation_rate
                        )
                    ]
                )

                mask_start = sum(self.savings_possible_actions)
                mask_end = mask_start + sum(self.mitigation_rate_possible_actions)
                mask[mask_start:mask_end] = mitigation_mask
            else:
                pass
            mask_dict[region_id] = mask

        return mask_dict


class OptimalMitigation(Rice):
    """Scenario where all agents mitigate to a given extent

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - maximum_mitigation_rate: the rate rate all agents will mitigate to.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="OptimateMitigation",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )
        self.maximum_mitigation_rate = 9

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()

            mitigation_mask = np.array(
                [0 for _ in range(self.maximum_mitigation_rate)]
                + [
                    1
                    for _ in range(
                        self.num_discrete_action_levels - self.maximum_mitigation_rate
                    )
                ]
            )

            mask_start = sum(self.savings_possible_actions)
            mask_end = mask_start + sum(self.mitigation_rate_possible_actions)
            mask[mask_start:mask_end] = mitigation_mask
            mask_dict[region_id] = mask

        return mask_dict


class MinimalMitigation(Rice):
    """Scenario where all agents mitigate to a given extent

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - maximum_mitigation_rate: the rate rate all agents will mitigate to.
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="MinimalMitigation",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )
        self.maximum_mitigation_rate = 1

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()

            mitigation_mask = np.array(
                [1 for _ in range(self.maximum_mitigation_rate)]
                + [
                    0
                    for _ in range(
                        self.num_discrete_action_levels - self.maximum_mitigation_rate
                    )
                ]
            )

            mask_start = sum(self.savings_possible_actions)
            mask_end = mask_start + sum(self.mitigation_rate_possible_actions)
            mask[mask_start:mask_end] = mitigation_mask
            mask_dict[region_id] = mask

        return mask_dict


class BasicClub(Rice):
    """Scenario where regions propose minimum mitigation rates, that are either accepted or rejected
    agents commit to the maximum of their accepted mitigation rates
    agents impose 0 tariff on club members
    outside club members get tariffed proportional to the diff between their tariffs.

        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario

        Attributes:
        - club_mitigation_rate: the rate rate all agents will mitigate to.
        - club_members: subset of states in club
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=True,  # If True then negotiation is on, else off
        scenario="BasicClub",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )

    def calc_possible_actions(self, action_type):
        if self.action_space_type == "discrete":
            if action_type == "savings":
                return [self.num_discrete_action_levels]
            if action_type == "mitigation_rate":
                return [self.num_discrete_action_levels]
            if action_type == "export_limit":
                return [self.num_discrete_action_levels]
            if action_type == "import_bids":
                return [self.num_discrete_action_levels] * self.num_regions
            if action_type == "import_tariffs":
                return [self.num_discrete_action_levels] * self.num_regions

            if action_type == "proposal":
                return [self.num_discrete_action_levels]

            if action_type == "proposal_decisions":
                return [2] * self.num_regions

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "export_limit":
            export_action_index = self.get_actions_index("export_limit")
            return [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_bids":
            tariffs_action_index = self.get_actions_index("import_bids")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposed_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")

            return [
                actions[region_id][proposal_actions_index_start]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposal_decisions":
            proposal_decisions_index_start = self.get_actions_index(
                "proposal_decisions"
            )
            num_evaluation_actions = len(self.evaluation_possible_actions)

            proposal_decisions = np.array(
                [
                    actions[region_id][
                        proposal_decisions_index_start : proposal_decisions_index_start
                        + num_evaluation_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )
            for region_id in range(self.num_regions):
                proposal_decisions[region_id, region_id] = 0

            return proposal_decisions

    def step_propose(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        proposed_mitigation_rates = self.get_actions(
            "proposed_mitigation_rate", actions
        )
        self.set_state("proposed_mitigation_rate", np.array(proposed_mitigation_rates))

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info

    def reset_state(self, key):

        if key == "proposed_mitigation_rate":
            self.set_state(key, value=np.zeros(self.num_regions))
        else:
            super().reset_state(key)

    def reset(self, *, seed=None, options=None):

        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state("timestep")
        self.reset_state("activity_timestep")

        # climate states
        self.reset_state("global_temperature")
        self.reset_state("global_carbon_mass")
        self.reset_state("global_exogenous_emissions")
        self.reset_state("global_land_emissions")
        self.reset_state("intensity_all_regions")
        self.reset_state("mitigation_rates_all_regions")

        # additional climate states for carbon and temperature model
        self.reset_state("global_alpha")
        self.reset_state("global_carbon_reservoirs")
        self.reset_state("global_cumulative_emissions")
        self.reset_state("global_cumulative_land_emissions")
        self.reset_state("global_emissions")
        self.reset_state("global_acc_pert_carb_stock")
        self.reset_state("global_temperature_boxes")

        # economic states
        self.reset_state("production_all_regions")
        self.reset_state("gross_output_all_regions")
        self.reset_state("aggregate_consumption")
        self.reset_state("investment_all_regions")
        self.reset_state("capital_all_regions")
        self.reset_state("capital_depreciation_all_regions")
        self.reset_state("labor_all_regions")
        self.reset_state("production_factor_all_regions")
        self.reset_state("current_balance_all_regions")
        self.reset_state("abatement_cost_all_regions")
        self.reset_state("mitigation_cost_all_regions")
        self.reset_state("damages_all_regions")
        self.reset_state("utility_all_regions")
        self.reset_state("social_welfare_all_regions")
        self.reset_state("reward_all_regions")

        # trade states
        self.reset_state("tariffs")
        self.reset_state("import_tariffs")
        self.reset_state("normalized_import_bids_all_regions")
        self.reset_state("import_bids_all_regions")
        self.reset_state("imports_minus_tariffs")
        self.reset_state("export_limit_all_regions")
        self.reset_state("export_regions_all_regions")

        # negotiation states
        self.reset_state("negotiation_stage")
        self.reset_state("savings_all_regions")
        self.reset_state("minimum_mitigation_rate_all_regions")
        self.reset_state("proposed_mitigation_rate")
        self.reset_state("promised_mitigation_rate")
        self.reset_state("requested_mitigation_rate")
        self.reset_state("proposal_decisions")

        info = {
            region: {} for region in range(self.num_regions)
        }  # for the new ray rllib env format
        return self.get_observations(), info

    def calc_mitigation_rate_lower_bound(self, region_id):

        # get all proposed_mitigation rates
        current_proposals = self.global_state["proposed_mitigation_rate"]["value"][
            self.current_timestep
        ]
        proposal_decisions = [
            self.global_state["proposal_decisions"]["value"][
                self.current_timestep, j, region_id
            ]
            for j in range(self.num_regions)
        ]

        # remove all rejected mitigation rates
        accepted_proposals = current_proposals * proposal_decisions

        # return max of accepted
        return max(accepted_proposals)

    def step_evaluate_proposals(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        proposal_decisions = self.get_actions("proposal_decisions", actions)

        self.set_state("proposal_decisions", proposal_decisions)

        for region_id in range(self.num_regions):
            min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)

            self.set_state(
                "minimum_mitigation_rate_all_regions", min_mitigation, region_id
            )

        observations = self.get_observations()

        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}
        return observations, rewards, terminateds, truncateds, info

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            if self.action_window:
                mask = self.calc_action_window(region_id)
            else:
                mask = self.default_agent_action_mask.copy()

            # minimum commitment
            min_mitigation_rate = int(
                self.get_state(
                    "minimum_mitigation_rate_all_regions",
                    region_id=region_id,
                    timestep=self.current_timestep,
                )
                * self.num_discrete_action_levels
            )

            current_mitigation_rate = int(
                self.get_state(
                    "mitigation_rates_all_regions",
                    region_id=region_id,
                    timestep=self.current_timestep,
                )
                * self.num_discrete_action_levels
            )

            # if agent has a minimum mitigation rate, it must increase mitigation until target reached
            if current_mitigation_rate < min_mitigation_rate:
                mitigation_mask = (
                    [0] * (current_mitigation_rate + 1)
                    + [1]
                    + [0]
                    * (self.num_discrete_action_levels - current_mitigation_rate - 2)
                )
            # if at the club level, agent has the possibility of keeping the same mitigation level
            elif current_mitigation_rate == min_mitigation_rate:
                mitigation_mask = (
                    [0] * (current_mitigation_rate)
                    + [1, 1]
                    + [0]
                    * (self.num_discrete_action_levels - current_mitigation_rate - 2)
                )
            # if above club level, normal action window applies
            elif current_mitigation_rate > min_mitigation_rate:
                pass

            mitigation_mask_start = sum(self.savings_possible_actions)
            mitigation_mask_end = mitigation_mask_start + sum(
                self.mitigation_rate_possible_actions
            )
            mask[mitigation_mask_start:mitigation_mask_end] = mitigation_mask

            # tariff non club members
            tariff_mask = []
            for other_region_id in range(self.num_regions):

                # get other regions mitigation commitment
                other_mitigation_rate = self.get_state(
                    "minimum_mitigation_rate_all_regions",
                    region_id=other_region_id,
                    timestep=self.current_timestep,
                )

                # if other region is self or in club
                if (other_region_id == region_id) or (
                    other_mitigation_rate >= min_mitigation_rate
                ):
                    # minimize tariff for free trade
                    regional_tariff_mask = [1] + [0] * (
                        self.num_discrete_action_levels - 1
                    )
                else:

                    # min tariff by difference between mitigation rate and club mitigation rate
                    tariff_rate = int(min_mitigation_rate - other_mitigation_rate)
                    regional_tariff_mask = [0] * tariff_rate + [1] * (
                        self.num_discrete_action_levels - tariff_rate
                    )
                tariff_mask.extend(regional_tariff_mask)

            # mask tariff
            tariffs_mask_start = self.get_actions_index("import_tariffs")
            tariff_mask_end = (
                self.num_regions * self.num_discrete_action_levels + tariffs_mask_start
            )
            mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask

        return mask_dict


class BasicClubFixedMembers(Rice):
    """Scenario where a subset of regions mitigate to a set amount,
    other agents get a tariff based on the difference between their mitigation rate
    and the club rate

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - club_mitigation_rate: the rate rate all agents will mitigate to.
    - club_members: subset of states in club
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="BasicClub",
    ):
        super().__init__(num_discrete_action_levels, negotiation_on, scenario)
        self.club_mitigation_rate = 9
        # Note: this will be updated later with more targeted region_ids
        self.club_members = [1, 2, 3, 4, 15, 8, 12]

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()

            # club members mitigate
            if region_id in self.club_members:
                mask = self.default_agent_action_mask.copy()
                # mask mitigation
                mitigation_mask = np.array(
                    [0 for _ in range(self.club_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels - self.club_mitigation_rate
                        )
                    ]
                )

                mitigation_mask_start = sum(self.savings_possible_actions)
                mitigation_mask_end = mitigation_mask_start + sum(
                    self.mitigation_rate_possible_actions
                )
                mask[mitigation_mask_start:mitigation_mask_end] = mitigation_mask

                # tariff non club members
                tariff_mask = []
                for other_region_id in range(self.num_regions):
                    # if other region is self or in club
                    if (other_region_id == region_id) or (
                        other_region_id in self.club_members
                    ):
                        # minimize tariff for free trade
                        regional_tariff_mask = [1] + [0] * (
                            self.num_discrete_action_levels - 1
                        )
                    else:
                        other_region_mitigation_rate = self.get_state(
                            "mitigation_rates_all_regions", region_id=other_region_id
                        )
                        # min tariff by difference between mitigation rate and club mitigation rate
                        tariff_rate = int(
                            self.club_mitigation_rate - other_region_mitigation_rate
                        )
                        regional_tariff_mask = [0] * tariff_rate + [1] * (
                            self.num_discrete_action_levels - tariff_rate
                        )
                    tariff_mask.extend(regional_tariff_mask)

                # mask tariff
                tariffs_mask_start = self.get_actions_index("import_tariffs")
                tariff_mask_end = (
                    self.num_regions * self.num_discrete_action_levels
                    + tariffs_mask_start
                )
                mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask

        return mask_dict


class ExportAction(Rice):
    """Scenario where a subset of regions mitigate to a set amount,
    other agents get a tariff based on the difference between their mitigation rate
    and the club rate

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - club_mitigation_rate: the rate rate all agents will mitigate to.
    - club_members: subset of states in club
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="ExportAction",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )

    def calc_possible_actions(self, action_type):
        if action_type == "savings":
            return [self.num_discrete_action_levels]
        if action_type == "mitigation_rate":
            return [self.num_discrete_action_levels]
        if action_type == "export_limit":
            return [self.num_discrete_action_levels]
        if action_type == "import_bids":
            return [self.num_discrete_action_levels] * self.num_regions
        if action_type == "import_tariffs":
            return [self.num_discrete_action_levels] * self.num_regions

        if action_type == "proposal":
            return [self.num_discrete_action_levels] * 2 * self.num_regions

        if action_type == "proposal_decisions":
            return [2] * self.num_regions

        if action_type == "export_regions":
            return [2] * self.num_regions

    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = (
            self.savings_possible_actions
            + self.mitigation_rate_possible_actions
            + self.export_limit_possible_actions
            + self.import_bids_possible_actions
            + self.import_tariff_possible_actions
            + self.export_regions_possible_actions
        )

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions + self.evaluation_possible_actions
            )

        return total_possible_actions

    def set_possible_actions(self):

        super().set_possible_actions()
        self.export_regions_possible_actions = self.calc_possible_actions(
            "export_regions"
        )

    def get_actions_index(self, action_type):

        if action_type == "export_regions":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
                + len(self.export_regions_possible_actions)
            )
        else:
            return super().get_actions_index(action_type)

    def get_actions(self, action_type, actions):

        if action_type == "export_regions":

            export_regions_index_start = self.get_actions_index("export_regions")
            num_export_regions_actions = len(self.export_regions_possible_actions)

            export_regions = np.array(
                [
                    actions[region_id][
                        export_regions_index_start : export_regions_index_start
                        + num_export_regions_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )

            for region_id in range(self.num_regions):
                export_regions[region_id, region_id] = 0
            return export_regions
        else:
            return super().get_actions(action_type, actions)

    def set_actions_in_global_state(self, actions_dict):
        for action_name, action_value in actions_dict.items():
            self.set_state(
                key=action_name,
                value=action_value,
                timestep=self.current_timestep,
                dtype=self.float_dtype,
            )

    def reset_state(self, key):

        if key != "export_regions_all_regions":
            super().reset_state(key)
        else:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)))

    def reset(self, *, seed=None, options=None):

        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state("timestep")
        self.reset_state("activity_timestep")

        # climate states
        self.reset_state("global_temperature")
        self.reset_state("global_carbon_mass")
        self.reset_state("global_exogenous_emissions")
        self.reset_state("global_land_emissions")
        self.reset_state("intensity_all_regions")
        self.reset_state("mitigation_rates_all_regions")

        # additional climate states for carbon model
        self.reset_state("global_alpha")
        self.reset_state("global_carbon_reservoirs")
        self.reset_state("global_cumulative_emissions")
        self.reset_state("global_cumulative_land_emissions")
        self.reset_state("global_emissions")
        self.reset_state("global_acc_pert_carb_stock")

        # economic states
        self.reset_state("production_all_regions")
        self.reset_state("gross_output_all_regions")
        self.reset_state("aggregate_consumption")
        self.reset_state("investment_all_regions")
        self.reset_state("capital_all_regions")
        self.reset_state("capital_depreciation_all_regions")
        self.reset_state("labor_all_regions")
        self.reset_state("production_factor_all_regions")
        self.reset_state("current_balance_all_regions")
        self.reset_state("abatement_cost_all_regions")
        self.reset_state("mitigation_cost_all_regions")
        self.reset_state("damages_all_regions")
        self.reset_state("utility_all_regions")
        self.reset_state("social_welfare_all_regions")
        self.reset_state("reward_all_regions")

        # trade states
        self.reset_state("tariffs")
        self.reset_state("import_tariffs")
        self.reset_state("normalized_import_bids_all_regions")
        self.reset_state("import_bids_all_regions")
        self.reset_state("imports_minus_tariffs")
        self.reset_state("export_limit_all_regions")
        self.reset_state("export_regions_all_regions")

        # negotiation states
        self.reset_state("negotiation_stage")
        self.reset_state("savings_all_regions")
        self.reset_state("minimum_mitigation_rate_all_regions")
        self.reset_state("promised_mitigation_rate")
        self.reset_state("requested_mitigation_rate")
        self.reset_state("proposal_decisions")

        info = {
            region: {} for region in range(self.num_regions)
        }  # for the new ray rllib env format
        return self.get_observations(), info

    def get_observations(self):
        """
        Format observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "timestep",
            "global_carbon_reservoirs",
            "global_cumulative_emissions",
            "global_cumulative_land_emissions",
            "global_alpha",
            "global_emissions",
            "global_acc_pert_carb_stock",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            "savings_all_regions",
            "mitigation_rates_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "export_regions_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]

        # Features concerning two regions
        bilateral_features = []

        if self.negotiation_on:
            global_features += ["negotiation_stage"]

            public_features += []

            private_features += [
                "minimum_mitigation_rate_all_regions",
            ]

            bilateral_features += [
                "promised_mitigation_rate",
                "requested_mitigation_rate",
                "proposal_decisions",
            ]

        shared_features = np.array([])
        for feature in global_features + public_features:
            shared_features = np.append(
                shared_features,
                self.flatten_array(
                    self.global_state[feature]["value"][self.current_timestep]
                    / self.global_state[feature]["norm"]
                ),
            )

        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for region_id in range(self.num_regions):

            # Add a region indicator array to the observation
            region_indicator = np.zeros(self.num_regions, dtype=self.float_dtype)
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert (
                    np.isnan(all_features).sum() == 0
                ), f"NaN in the features: {feature}"
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            for feature in bilateral_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert self.global_state[feature]["value"].shape[2] == self.num_regions
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, :, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            features_dict[region_id] = all_features

        # Fetch the action mask dictionary, keyed by region_id.
        action_mask_dict = self.calc_action_mask()

        # Form the observation dictionary keyed by region id.
        obs_dict = {}
        for region_id in range(self.num_regions):
            obs_dict[region_id] = {
                _FEATURES: features_dict[region_id],
                _ACTION_MASK: action_mask_dict[region_id],
            }

        return obs_dict

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            open_for_trade = self.get_importable_regions(region_id)

            imports_mask = []

            for other_region in range(self.num_regions):
                if other_region != region_id:
                    if open_for_trade[other_region] == 1:
                        imports_mask.extend([1] * self.num_discrete_action_levels)
                    else:
                        imports_mask.extend(
                            [1] + [0] * (self.num_discrete_action_levels - 1)
                        )
                else:
                    imports_mask.extend(
                        [1] + [0] * (self.num_discrete_action_levels - 1)
                    )
            mask_dict[region_id] = mask

            mask_start = sum(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
            )

            mask_end = mask_start + sum(self.calc_possible_actions("import_bids"))
            mask[mask_start:mask_end] = np.array(imports_mask)

        return mask_dict

    def get_importable_regions(self, region_id):
        """
        Get the output of the export_region action for a given region
        """

        export_regions = self.get_state("export_regions_all_regions")
        open_for_trade = export_regions[:, region_id]
        return open_for_trade

    def step_climate_and_economy(self, actions=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)

        actions_dict = {
            "savings_all_regions": self.get_actions("savings", actions),
            "mitigation_rates_all_regions": self.get_actions(
                "mitigation_rate", actions
            ),
            "export_limit_all_regions": self.get_actions("export_limit", actions),
            "import_bids_all_regions": self.get_actions("import_bids", actions),
            "import_tariffs_all_regions": self.get_actions("import_tariffs", actions),
            "export_regions_all_regions": self.get_actions("export_regions", actions),
        }

        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(
            gross_outputs, gross_imports, net_imports
        )
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)
        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = {}

        return observations, rewards, terminateds, truncateds, info


class Custom_1(Rice):
    """Scenario

    Arguments:
    - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
    - negotiation_on (boolean): whether negotiation actions are available to agents
    - scenario (str): name of scenario

    Attributes:
    - club_mitigation_rate: the rate rate all agents will mitigate to.
    - club_members: subset of states in club
    """

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        scenario="ExportAction",
        action_space_type="discrete",  # or "continuous"
        dmg_function="base",
        carbon_model="base",
        temperature_calibration="base",
        prescribed_emissions=None,
    ):
        super().__init__(
            negotiation_on=negotiation_on,  # If True then negotiation is on, else off
            scenario=scenario,
            num_discrete_action_levels=num_discrete_action_levels,
            action_space_type=action_space_type,  # or "continuous"
            dmg_function=dmg_function,
            carbon_model=carbon_model,
            temperature_calibration=temperature_calibration,
            prescribed_emissions=prescribed_emissions,
        )

    def calc_possible_actions(self, action_type):
        if action_type == "savings":
            return [self.num_discrete_action_levels]
        if action_type == "mitigation_rate":
            return [self.num_discrete_action_levels]
        if action_type == "export_limit":
            return [self.num_discrete_action_levels]
        if action_type == "import_bids":
            return [self.num_discrete_action_levels] * self.num_regions
        if action_type == "import_tariffs":
            return [self.num_discrete_action_levels] * self.num_regions

        if action_type == "proposal":
            return [self.num_discrete_action_levels] * 2 * self.num_regions

        if action_type == "proposal_decisions":
            return [2] * self.num_regions

        if action_type == "export_regions":
            return [2] * self.num_regions

    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = (
            self.savings_possible_actions
            + self.mitigation_rate_possible_actions
            + self.export_limit_possible_actions
            + self.import_bids_possible_actions
            + self.import_tariff_possible_actions
            + self.export_regions_possible_actions
        )

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions + self.evaluation_possible_actions
            )

        return total_possible_actions

    def set_possible_actions(self):

        super().set_possible_actions()
        self.export_regions_possible_actions = self.calc_possible_actions(
            "export_regions"
        )

    def get_actions_index(self, action_type):

        if action_type == "export_regions":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
                + len(self.export_regions_possible_actions)
            )
        else:
            return super().get_actions_index(action_type)

    def get_actions(self, action_type, actions):

        if action_type == "export_regions":

            export_regions_index_start = self.get_actions_index("export_regions")
            num_export_regions_actions = len(self.export_regions_possible_actions)

            export_regions = np.array(
                [
                    actions[region_id][
                        export_regions_index_start : export_regions_index_start
                        + num_export_regions_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )

            for region_id in range(self.num_regions):
                export_regions[region_id, region_id] = 0
            return export_regions
        else:
            return super().get_actions(action_type, actions)

    def set_actions_in_global_state(self, actions_dict):
        for action_name, action_value in actions_dict.items():
            self.set_state(
                key=action_name,
                value=action_value,
                timestep=self.current_timestep,
                dtype=self.float_dtype,
            )

    def reset_state(self, key):

        if key != "export_regions_all_regions":
            super().reset_state(key)
        else:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)))

    def reset(self, *, seed=None, options=None):

        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state("timestep")
        self.reset_state("activity_timestep")

        # climate states
        self.reset_state("global_temperature")
        self.reset_state("global_carbon_mass")
        self.reset_state("global_exogenous_emissions")
        self.reset_state("global_land_emissions")
        self.reset_state("intensity_all_regions")
        self.reset_state("mitigation_rates_all_regions")

        # additional climate states for carbon model
        self.reset_state("global_alpha")
        self.reset_state("global_carbon_reservoirs")
        self.reset_state("global_cumulative_emissions")
        self.reset_state("global_cumulative_land_emissions")
        self.reset_state("global_emissions")
        self.reset_state("global_acc_pert_carb_stock")

        # economic states
        self.reset_state("production_all_regions")
        self.reset_state("gross_output_all_regions")
        self.reset_state("aggregate_consumption")
        self.reset_state("investment_all_regions")
        self.reset_state("capital_all_regions")
        self.reset_state("capital_depreciation_all_regions")
        self.reset_state("labor_all_regions")
        self.reset_state("production_factor_all_regions")
        self.reset_state("current_balance_all_regions")
        self.reset_state("abatement_cost_all_regions")
        self.reset_state("mitigation_cost_all_regions")
        self.reset_state("damages_all_regions")
        self.reset_state("utility_all_regions")
        self.reset_state("social_welfare_all_regions")
        self.reset_state("reward_all_regions")

        # trade states
        self.reset_state("tariffs")
        self.reset_state("import_tariffs")
        self.reset_state("normalized_import_bids_all_regions")
        self.reset_state("import_bids_all_regions")
        self.reset_state("imports_minus_tariffs")
        self.reset_state("export_limit_all_regions")
        self.reset_state("export_regions_all_regions")

        # negotiation states
        self.reset_state("negotiation_stage")
        self.reset_state("savings_all_regions")
        self.reset_state("minimum_mitigation_rate_all_regions")
        self.reset_state("promised_mitigation_rate")
        self.reset_state("requested_mitigation_rate")
        self.reset_state("proposal_decisions")

        info = {
            region: {} for region in range(self.num_regions)
        }  # for the new ray rllib env format
        return self.get_observations(), info

    def get_observations(self):
        """
        Format observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "timestep",
            "global_carbon_reservoirs",
            "global_cumulative_emissions",
            "global_cumulative_land_emissions",
            "global_alpha",
            "global_emissions",
            "global_acc_pert_carb_stock",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            "savings_all_regions",
            "mitigation_rates_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "export_regions_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]

        # Features concerning two regions
        bilateral_features = []

        if self.negotiation_on:
            global_features += ["negotiation_stage"]

            public_features += []

            private_features += [
                "minimum_mitigation_rate_all_regions",
            ]

            bilateral_features += [
                "promised_mitigation_rate",
                "requested_mitigation_rate",
                "proposal_decisions",
            ]

        shared_features = np.array([])
        for feature in global_features + public_features:
            shared_features = np.append(
                shared_features,
                self.flatten_array(
                    self.global_state[feature]["value"][self.current_timestep]
                    / self.global_state[feature]["norm"]
                ),
            )

        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for region_id in range(self.num_regions):

            # Add a region indicator array to the observation
            region_indicator = np.zeros(self.num_regions, dtype=self.float_dtype)
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert (
                    np.isnan(all_features).sum() == 0
                ), f"NaN in the features: {feature}"
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            for feature in bilateral_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert self.global_state[feature]["value"].shape[2] == self.num_regions
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, :, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            features_dict[region_id] = all_features

        # Fetch the action mask dictionary, keyed by region_id.
        action_mask_dict = self.calc_action_mask()

        # Form the observation dictionary keyed by region id.
        obs_dict = {}
        for region_id in range(self.num_regions):
            obs_dict[region_id] = {
                _FEATURES: features_dict[region_id],
                _ACTION_MASK: action_mask_dict[region_id],
            }

        return obs_dict

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            open_for_trade = self.get_importable_regions(region_id)

            imports_mask = []

            for other_region in range(self.num_regions):
                if other_region != region_id:
                    if open_for_trade[other_region] == 1:
                        imports_mask.extend([1] * self.num_discrete_action_levels)
                    else:
                        imports_mask.extend(
                            [1] + [0] * (self.num_discrete_action_levels - 1)
                        )
                else:
                    imports_mask.extend(
                        [1] + [0] * (self.num_discrete_action_levels - 1)
                    )
            mask_dict[region_id] = mask

            mask_start = sum(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
            )

            mask_end = mask_start + sum(self.calc_possible_actions("import_bids"))
            mask[mask_start:mask_end] = np.array(imports_mask)

        return mask_dict

    def get_importable_regions(self, region_id):
        """
        Get the output of the export_region action for a given region
        """

        export_regions = self.get_state("export_regions_all_regions")
        open_for_trade = export_regions[:, region_id]
        return open_for_trade

    def step_climate_and_economy(self, actions=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)

        actions_dict = {
            "savings_all_regions": self.get_actions("savings", actions),
            "mitigation_rates_all_regions": self.get_actions(
                "mitigation_rate", actions
            ),
            "export_limit_all_regions": self.get_actions("export_limit", actions),
            "import_bids_all_regions": self.get_actions("import_bids", actions),
            "import_tariffs_all_regions": self.get_actions("import_tariffs", actions),
            "export_regions_all_regions": self.get_actions("export_regions", actions),
        }

        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(
            actions_dict["mitigation_rates_all_regions"]
        )
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(
            gross_outputs, actions_dict["savings_all_regions"]
        )

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(
            actions_dict["import_bids_all_regions"],
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(
            gross_outputs, gross_imports, net_imports
        )
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(consumptions)
        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = {}

        return observations, rewards, terminateds, truncateds, info
