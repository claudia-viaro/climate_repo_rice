# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Custom Pytorch policy models to use with RLlib.
"""

# API reference:
# https://docs.ray.io/en/latest/rllib/rllib-models.html#custom-pytorch-models

import numpy as np
import os
import json
from gymnasium.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override
import logging

torch, nn = try_import_torch()

_ACTION_MASK = "action_mask"


class TorchLinear(TorchModelV2, nn.Module):
    """
    Fully-connected Pytorch policy model.
    """

    custom_name = "torch_linear_discrete"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_dims=None,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if fc_dims is None:
            fc_dims = [256, 256]

        # Check Observation spaces
        self.rollout_step_counter = 0

        self.observation_space = obs_space.original_space
        # print("self.observation_space", self.observation_space)
        if not isinstance(self.observation_space, Dict):
            if isinstance(self.observation_space, Box):
                raise TypeError(
                    "({name}) Observation space should be a gym Dict. "
                    "Is a Box of shape {self.observation_space.shape}"
                )
            raise TypeError(
                f"({name}) Observation space should be a gym Dict. "
                "Is {type(self.observation_space))} instead."
            )

        flattened_obs_size = self.get_flattened_obs_size()
        # print("sflattened_obs_size", flattened_obs_size)
        # Model only outputs policy logits,
        # values are accessed via the self.value_function
        self.values = None

        num_fc_layers = len(fc_dims)

        input_dims = [flattened_obs_size] + fc_dims[:-1]
        # print("input_dims", input_dims)
        output_dims = fc_dims

        self.fc_dict = nn.ModuleDict()
        for fc_layer in range(num_fc_layers):
            self.fc_dict[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )

        # policy network (list of heads)
        policy_heads = [None for _ in range(len(action_space))]
        self.output_dims = []  # Network output dimension(s)

        for idx, act_space in enumerate(action_space):
            output_dim = act_space.n
            self.output_dims += [output_dim]
            policy_heads[idx] = nn.Linear(fc_dims[-1], output_dim)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        self.vf_head = nn.Linear(fc_dims[-1], 1)

        # used for action masking
        self.action_mask = None

    def get_flattened_obs_size(self):
        """Get the total size of the observation after flattening."""
        if isinstance(self.observation_space, Box):
            obs_size = np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_size = 0
            for key in sorted(self.observation_space):
                if key == _ACTION_MASK:
                    pass
                else:
                    obs_size += np.prod(self.observation_space[key].shape)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return int(obs_size)

    def get_flattened_obs(self, obs):
        """Get the flattened observation (ignore the action masks)."""
        if isinstance(self.observation_space, Box):
            return self.reshape_and_flatten(obs)
        if isinstance(self.observation_space, Dict):
            flattened_obs_dict = {}
            for key in sorted(self.observation_space):
                assert key in obs
                if key == _ACTION_MASK:
                    self.action_mask = self.reshape_and_flatten_obs(obs[key])
                else:
                    flattened_obs_dict[key] = self.reshape_and_flatten_obs(obs[key])
            flattened_obs = torch.cat(list(flattened_obs_dict.values()), dim=-1)
            return flattened_obs
        raise NotImplementedError("Observation space must be of Box or Dict type")

    def summarize_tensor(self, tensor, name="", max_items=5):
        arr = tensor.detach().cpu().numpy()
        summary = f"{name} shape: {arr.shape}, values: {arr.flatten()[:max_items]}"
        return summary

    def compute_action_head_gradients(self, input_dict):
        """
        Forward + backward pass to compute gradients per policy head.
        Use this outside the RLlib loop, e.g. for debugging/analysis.
        """
        self.zero_grad()

        # Unwrap observation
        original_obs = restore_original_dimensions(
            input_dict["obs"], self.obs_space.original_space, "torch"
        )
        obs = self.get_flattened_obs(original_obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # Forward pass through shared FC layers
        for layer in self.fc_dict:
            obs = self.fc_dict[layer](obs)

        logits_per_head = [head(obs) for head in self.policy_head]
        grad_info = []

        for i, logits in enumerate(logits_per_head):
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            try:
                logp = dist.log_prob(action)
            except Exception as e:
                print(
                    f"Failed to compute log_prob. Action shape: {action.shape}, logits: {logits.shape}"
                )
                raise

            loss = -logp.mean()  # use dummy advantage=1.0
            loss.backward(retain_graph=True)

            grad_norms = {}
            for name, param in self.policy_head[i].named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            grad_info.append((f"head_{i}", grad_norms))
            self.zero_grad()  # Reset before next head

        return grad_info

    @staticmethod
    def reshape_and_flatten_obs(obs):
        """Flatten observation."""
        assert len(obs.shape) >= 2
        batch_dim = obs.shape[0]
        return obs.reshape(batch_dim, -1)

    @override(TorchModelV2)
    def value_function(self):
        """Returns the estimated value function."""
        return self.values.reshape(-1)

    @staticmethod
    def apply_logit_mask(logits, mask):
        """
        Mask values of 1 are valid actions.
        Add huge negative values to logits with 0 mask values.
        """
        logit_mask = torch.ones_like(logits) * -10000000
        logit_mask = logit_mask * (1 - mask)
        return logits + logit_mask

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        print("Input dict keys:", input_dict.keys()) # Input dict keys: dict_keys(['obs', 'eps_id', 'agent_index', 'obs_flat'])
        if "agent_index" in input_dict:
            print("Agent IDs:", input_dict["agent_index"]) # Agent IDs: tensor([0, 1, 2])
        elif "infos" in input_dict:
            print("Infos keys:", input_dict["infos"].keys())
        """
        """You should implement forward() of forward_rnn() in your subclass."""
        is_training = input_dict.is_training
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        # Note: restoring original obs
        # as RLlib does not seem to be doing it automatically!
        original_obs = restore_original_dimensions(
            input_dict["obs"], self.obs_space.original_space, "torch"
        )
        # print("original_obs", original_obs)
        obs = self.get_flattened_obs(original_obs)

        # Feed through the FC layers
        for layer in range(len(self.fc_dict)):
            output = self.fc_dict[str(layer)](obs)
            obs = output
        logits = output
        self.unmasked_savings_logits = self.policy_head[0](logits)  ### change back!
        # Compute the action probabilities and the value function estimate
        # Apply action mask to the logits as well.
        action_masks = [None for _ in range(len(self.output_dims))]
        if self.action_mask is not None:
            start = 0
            for idx, dim in enumerate(self.output_dims):
                action_masks[idx] = self.action_mask[..., start : start + dim]
                start = start + dim
        action_logits = [
            self.apply_logit_mask(ph(logits), action_masks[idx])
            for idx, ph in enumerate(self.policy_head)
        ]
        self.values = self.vf_head(logits)[..., 0]

        concatenated_action_logits = torch.cat(action_logits, dim=-1)
        # Optionally, save to file for deeper analysis
        """
        print(
            "[DEBUG] Action logits before masking:\n"
            + "\n".join(
                [
                    self.summarize_tensor(ph(logits), f"  Head {i}")
                    for i, ph in enumerate(self.policy_head)
                ]
            )
        )

        print(
            "[DEBUG] Action masks:\n"
            + "\n".join(
                [
                    (
                        self.summarize_tensor(am, f"  Mask {i}")
                        if am is not None
                        else f"  Mask {i}: None"
                    )
                    for i, am in enumerate(action_masks)
                ]
            )
        )

        print(
            "[DEBUG] Masked logits:\n"
            + "\n".join(
                [
                    self.summarize_tensor(al, f"  Masked Head {i}")
                    for i, al in enumerate(action_logits)
                ]
            )
        )
        """

        """ this is really too heavy to save!"""

        """
        if not input_dict.get("is_training", False):
            batch_size = logits.shape[0]  # usually 32

            episode_ids = input_dict.get("eps_id", [None] * batch_size)

            with open(self.log_file, "a") as f:
                for i in range(batch_size):
                    timestep = self.rollout_step_counter + i
                    for head_idx, masked_logit in enumerate(action_logits):
                        head_logit = masked_logit[i].detach().cpu().numpy().tolist()

                        log_entry = {
                            "timestep": timestep,
                            "episode_id": (
                                episode_ids[i].item()
                                if episode_ids[i] is not None
                                else None
                            ),
                            "head": head_idx,
                            "masked_logits": head_logit,
                        }

                        f.write(json.dumps(log_entry) + "\n")

            self.rollout_step_counter += batch_size
        """
        return (
            torch.reshape(concatenated_action_logits, [-1, self.num_outputs]),
            state,
        )


ModelCatalog.register_custom_model("torch_linear_discrete", TorchLinear)
