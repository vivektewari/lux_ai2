import gym
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

from .in_blocks import DictInputLayer
from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.act_spaces import MAX_OVERLAPPING_ACTIONS
from ..lux_gym.reward_spaces import RewardSpec


class DictActor(nn.Module):
    def __init__(
            self,
            in_channels: int,
            action_space: gym.spaces.Dict,
    ):
        super(DictActor, self).__init__()
        if not all([isinstance(space, gym.spaces.MultiDiscrete) for space in action_space.spaces.values()]):
            act_space_types = {key: type(space) for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must be MultiDiscrete. Found: {act_space_types}")
        if not all([len(space.shape) == 4 for space in action_space.spaces.values()]):
            act_space_ndims = {key: space.shape for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must have 4 dimensions. Found: {act_space_ndims}")
        if not all([space.nvec.min() == space.nvec.max() for space in action_space.spaces.values()]):
            act_space_n_acts = {key: np.unique(space.nvec) for key, space in action_space.spaces.items()}
            raise ValueError(f"Each action space must have the same number of actions throughout the space. "
                             f"Found: {act_space_n_acts}")
        self.n_actions = {
            key: space.nvec.max() for key, space in action_space.spaces.items()
        }
        # An action plane shape usually takes the form (n,), where n >= 1 and is used when multiple stacked units
        # must output different actions.
        self.action_plane_shapes = {
            key: space.shape[:-3] for key, space in action_space.spaces.items()
        }
        assert all([len(aps) == 1 for aps in self.action_plane_shapes.values()])
        self.actors = nn.ModuleDict({
            key: nn.Conv2d(
                in_channels,
                n_act * np.prod(self.action_plane_shapes[key]),
                (1, 1)
            ) for key, n_act in self.n_actions.items()
        })

    def forward(
            self,
            x: torch.Tensor,
            available_actions_mask: Dict[str, torch.Tensor],
            sample: bool,
            actions_per_square: Optional[int] = MAX_OVERLAPPING_ACTIONS
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Expects an input of shape batch_size * 2, n_channels, h, w
        This input will be projected by the actors, and then converted to shape batch_size, n_channels, 2, h, w
        """
        policy_logits_out = {}
        actions_out = {}
        b, _, h, w = x.shape
        for key, actor in self.actors.items():
            n_actions = self.n_actions[key]
            action_plane_shape = self.action_plane_shapes[key]
            logits = actor(x).view(b // 2, 2, n_actions, *action_plane_shape, h, w)
            # Move the logits dimension to the end and swap the player and channel dimensions
            logits = logits.permute(0, 3, 1, 4, 5, 2).contiguous()
            # In case all actions are masked, unmask all actions
            # We first have to cast it to an int tensor to avoid errors in kaggle environment
            aam = available_actions_mask[key]
            orig_dtype = aam.dtype
            aam_new_type = aam.to(dtype=torch.int64)
            aam_filled = torch.where(
                (~aam).all(dim=-1, keepdim=True),
                torch.ones_like(aam_new_type),
                aam_new_type.to(dtype=torch.int64)
            ).to(orig_dtype)
            assert logits.shape == aam_filled.shape
            logits = logits + torch.where(
                aam_filled,
                torch.zeros_like(logits),
                torch.zeros_like(logits) + float("-inf")
            )
            actions = DictActor.logits_to_actions(logits.view(-1, n_actions), sample, actions_per_square)
            policy_logits_out[key] = logits
            actions_out[key] = actions.view(*logits.shape[:-1], -1)
        return policy_logits_out, actions_out

    @staticmethod
    @torch.no_grad()
    def logits_to_actions(logits: torch.Tensor, sample: bool, actions_per_square: Optional[int]) -> torch.Tensor:
        if actions_per_square is None:
            actions_per_square = logits.shape[-1]
        if sample:
            probs = F.softmax(logits, dim=-1)
            # In case there are fewer than MAX_OVERLAPPING_ACTIONS available actions, we add a small eps value
            probs = torch.where(
                (probs > 0.).sum(dim=-1, keepdim=True) >= actions_per_square,
                probs,
                probs + 1e-10
            )
            return torch.multinomial(
                probs,
                num_samples=min(actions_per_square, probs.shape[-1]),
                replacement=False
            )
        else:
            return logits.argsort(dim=-1, descending=True)[..., :actions_per_square]

