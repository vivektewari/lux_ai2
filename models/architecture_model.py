from torch import nn
import gym
class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            base_out_channels: int,
            action_space: gym.spaces.Dict,
            reward_space: RewardSpec,
            actor_critic_activation: Callable = nn.ReLU,
            n_action_value_layers: int = 2,
            n_value_heads: int = 1,
            rescale_value_input: bool = True
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.dict_input_layer = DictInputLayer()
        self.base_model = base_model
        self.base_out_channels = base_out_channels

        if n_action_value_layers < 2:
            raise ValueError("n_action_value_layers must be >= 2 in order to use spectral_norm")

        """
        actor_layers = []
        baseline_layers = []
        for i in range(n_action_value_layers - 1):
            actor_layers.append(
                nn.utils.spectral_norm(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            )
            actor_layers.append(actor_critic_activation())
            baseline_layers.append(
                nn.utils.spectral_norm(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            )
            baseline_layers.append(actor_critic_activation())
        self.actor_base = nn.Sequential(*actor_layers)
        self.actor = DictActor(self.base_out_channels, action_space)
        self.baseline_base = nn.Sequential(*baseline_layers)"""

        self.actor_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )
        self.actor = DictActor(self.base_out_channels, action_space)

        self.baseline_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )
        self.baseline = BaselineLayer(
            in_channels=self.base_out_channels,
            reward_space=reward_space,
            n_value_heads=n_value_heads,
            rescale_input=rescale_value_input
        )

    def forward(
            self,
            x: Dict[str, Union[dict, torch.Tensor]],
            sample: bool = True,
            **actor_kwargs
    ) -> Dict[str, Any]:
        x, input_mask, available_actions_mask, subtask_embeddings = self.dict_input_layer(x)
        base_out, input_mask = self.base_model((x, input_mask))
        if subtask_embeddings is not None:
            subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)
        policy_logits, actions = self.actor(
            self.actor_base(base_out),
            available_actions_mask=available_actions_mask,
            sample=sample,
            **actor_kwargs
        )
        baseline = self.baseline(self.baseline_base(base_out), input_mask, subtask_embeddings)
        return dict(
            actions=actions,
            policy_logits=policy_logits,
            baseline=baseline
        )

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)

    @staticmethod
    def make_spectral_norm_head_base(n_layers: int, n_channels: int, activation: Callable) -> nn.Module:
        """
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
        assert n_layers >= 2
        layers = []
        for i in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
            layers.append(activation())
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1)))
        )
        layers.append(activation())

        return nn.Sequential(*layers)