"""
Custom ActorCriticPolicy with separate value networks for offense and defense.

This addresses the fundamental issue where a single centralized critic cannot properly
learn value functions for both teams in a zero-sum self-play setting with symmetric rewards.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule


class DualCriticActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with separate value networks for offensive and defensive perspectives.
    
    The actor (policy network) remains unified and conditions on role_flag.
    The critics are split:
    - value_net_offense: predicts returns when playing as offense (role_flag=1.0)
    - value_net_defense: predicts returns when playing as defense (role_flag=-1.0)
    
    This resolves the issue where a single critic tries to predict both positive (offense)
    and negative (defense) returns for the same state, leading to poor value estimates.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Initialize parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Build the actor and TWO critic networks.
        
        Overrides the parent method to create separate value networks for offense and defense.
        """
        # Build the standard actor network (policy)
        super()._build_mlp_extractor()
        
        # Now replace the single value_net with two separate networks
        # Get the latent dimension from the existing mlp_extractor
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        
        # Create separate value networks
        self.value_net_offense = nn.Linear(latent_dim_vf, 1)
        self.value_net_defense = nn.Linear(latent_dim_vf, 1)
        
        # Initialize with orthogonal initialization if requested
        if self.ortho_init:
            # Robustly initialize all Linear layers in each value network
            # Works for both single Linear and Sequential/multi-layer networks
            for net, gain in [(self.value_net_offense, 1.0), (self.value_net_defense, 1.0)]:
                # If it's a single Linear layer
                if isinstance(net, nn.Linear):
                    nn.init.orthogonal_(net.weight, gain=gain)
                    nn.init.constant_(net.bias, 0)
                # If it's a Sequential or Module with submodules, recursively init all Linear layers
                else:
                    for module in net.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.orthogonal_(module.weight, gain=gain)
                            nn.init.constant_(module.bias, 0)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critics).
        
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value (using appropriate critic), log probability of the action
        """
        # Extract role_flag BEFORE feature extraction
        role_flags = self._extract_role_flag(obs)
        
        # Preprocess observation if needed
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        # Evaluate the values for the given observations
        # Determine which value network to use based on role_flag
        values = self._get_value_from_latent(latent_vf, role_flags)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def _extract_role_flag(self, obs):
        """
        Extract role_flag from observation before it gets preprocessed.
        
        :param obs: Original observation (dict or tensor)
        :return: role_flag tensor of shape (batch_size, 1)
        """
        if isinstance(obs, dict) and "role_flag" in obs:
            role_flags = obs["role_flag"]
            # Ensure it's a tensor
            if not isinstance(role_flags, torch.Tensor):
                role_flags = torch.as_tensor(role_flags, dtype=torch.float32)
            # Ensure shape is (batch_size, 1)
            if role_flags.dim() == 1:
                role_flags = role_flags.unsqueeze(-1)
            return role_flags
        else:
            # Fallback: assume offense (this shouldn't happen in normal training)
            print("[WARNING] Could not extract role_flag from observation, defaulting to offense")
            batch_size = 1  # We don't know batch size yet
            return torch.ones((batch_size, 1), dtype=torch.float32)
    
    def _get_value_from_latent(self, latent_vf: torch.Tensor, role_flags: torch.Tensor) -> torch.Tensor:
        """
        Get the value estimate using the appropriate critic network based on role_flag.
        
        :param latent_vf: Latent value features
        :param role_flags: Pre-extracted role flags (batch_size, 1)
        :return: Value estimate
        """
        # Get values from both networks
        values_offense = self.value_net_offense(latent_vf)
        values_defense = self.value_net_defense(latent_vf)
        
        # Select appropriate value based on role_flag
        # role_flag=1.0 -> use offense value, role_flag=-1.0 -> use defense value
        # Ensure role_flags is on the same device
        role_flags = role_flags.to(latent_vf.device)
        is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1)  # (batch_size, 1)
        values = torch.where(is_offense, values_offense, values_defense)
        
        return values

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        Uses the appropriate critic network based on role_flag.
        
        :param obs: Observation
        :return: the estimated values.
        """
        # Extract role_flag BEFORE feature extraction
        role_flags = self._extract_role_flag(obs)
        
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        return self._get_value_from_latent(latent_vf, role_flags)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy, using the appropriate value network.
        
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution.
        """
        # Extract role_flag BEFORE feature extraction
        role_flags = self._extract_role_flag(obs)
        
        # Preprocess observation if needed
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self._get_value_from_latent(latent_vf, role_flags)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

