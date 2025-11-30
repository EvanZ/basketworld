"""
Custom ActorCriticPolicy with separate value networks for offense and defense,
and optionally separate action networks (dual policy) for offense and defense.

This addresses the fundamental issue where a single centralized critic cannot properly
learn value functions for both teams in a zero-sum self-play setting with symmetric rewards.

With dual policy enabled, the actor networks are also split to allow offense and defense
to learn fundamentally different action selection strategies.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule


class DualCriticActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    Actor-Critic policy with separate value networks for offensive and defensive perspectives,
    and optionally separate action networks for offense and defense.
    
    The critics are always split:
    - value_net_offense: predicts returns when playing as offense (role_flag=1.0)
    - value_net_defense: predicts returns when playing as defense (role_flag=-1.0)
    
    When use_dual_policy=True, the actors are also split:
    - action_net_offense: action logits when playing as offense
    - action_net_defense: action logits when playing as defense
    
    This resolves the issue where a single critic tries to predict both positive (offense)
    and negative (defense) returns for the same state, and where a single actor may be
    biased toward one role's behavior.
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
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_dual_policy: bool = False,
    ):
        # Store use_dual_policy BEFORE calling super().__init__ because
        # _build_mlp_extractor is called during parent initialization
        self.use_dual_policy = use_dual_policy
        
        # Temporary storage for role_flags during forward pass
        # This allows _get_action_dist_from_latent to access role_flags
        # without changing its method signature
        self._current_role_flags: Optional[torch.Tensor] = None
        
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
        Build the actor and TWO critic networks, and optionally TWO action networks.
        
        Overrides the parent method to create separate value networks for offense and defense,
        and optionally separate action networks.
        """
        # Build the standard networks (actor, critic, mlp_extractor)
        super()._build_mlp_extractor()
        
        # Now replace the single value_net with two separate networks
        # Get the latent dimension from the existing mlp_extractor
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        
        # Create separate value networks
        self.value_net_offense = nn.Linear(latent_dim_vf, 1)
        self.value_net_defense = nn.Linear(latent_dim_vf, 1)
        
        # Optionally create separate action networks
        if self.use_dual_policy:
            latent_dim_pi = self.mlp_extractor.latent_dim_pi
            # Calculate action dimension from the action space
            # For MultiDiscrete, this is the sum of all action dimensions
            # Note: self.action_net doesn't exist yet (created after _build_mlp_extractor returns)
            if hasattr(self.action_space, 'nvec'):
                # MultiDiscrete action space
                action_dim = int(sum(self.action_space.nvec))
            elif hasattr(self.action_space, 'n'):
                # Discrete action space
                action_dim = int(self.action_space.n)
            else:
                # Fallback: try to get shape
                action_dim = int(self.action_space.shape[0]) if self.action_space.shape else 1
            
            self.action_net_offense = nn.Linear(latent_dim_pi, action_dim)
            self.action_net_defense = nn.Linear(latent_dim_pi, action_dim)
        
        # Initialize with orthogonal initialization if requested
        if self.ortho_init:
            # Initialize value networks
            for net, gain in [(self.value_net_offense, 1.0), (self.value_net_defense, 1.0)]:
                if isinstance(net, nn.Linear):
                    nn.init.orthogonal_(net.weight, gain=gain)
                    nn.init.constant_(net.bias, 0)
                else:
                    for module in net.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.orthogonal_(module.weight, gain=gain)
                            nn.init.constant_(module.bias, 0)
            
            # Initialize action networks if dual policy
            if self.use_dual_policy:
                # Use gain=0.01 for action networks (same as SB3 default for action_net)
                for net in [self.action_net_offense, self.action_net_defense]:
                    if isinstance(net, nn.Linear):
                        nn.init.orthogonal_(net.weight, gain=0.01)
                        nn.init.constant_(net.bias, 0)
                    else:
                        for module in net.modules():
                            if isinstance(module, nn.Linear):
                                nn.init.orthogonal_(module.weight, gain=0.01)
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
        
        # Store role_flags for use in _get_action_dist_from_latent
        self._current_role_flags = role_flags
        
        try:
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
        finally:
            # Clear stored role_flags
            self._current_role_flags = None

    def _extract_role_flag(self, obs):
        """
        Extract role_flag from observation before it gets preprocessed.
        
        :param obs: Original observation (dict, TensorDict, or tensor)
        :return: role_flag tensor of shape (batch_size, 1)
        """
        # Handle dict-like objects (dict, TensorDict, etc.)
        # Use duck typing instead of isinstance(obs, dict) to handle
        # SB3's TensorDict and other dict-like types after obs_to_tensor()
        try:
            if hasattr(obs, '__getitem__') and hasattr(obs, 'keys') and "role_flag" in obs:
                role_flags = obs["role_flag"]
                # Ensure it's a tensor
                if not isinstance(role_flags, torch.Tensor):
                    role_flags = torch.as_tensor(role_flags, dtype=torch.float32)
                # Ensure shape is (batch_size, 1)
                if role_flags.dim() == 0:
                    role_flags = role_flags.unsqueeze(0).unsqueeze(-1)
                elif role_flags.dim() == 1:
                    role_flags = role_flags.unsqueeze(-1)
                return role_flags
        except (KeyError, TypeError, AttributeError):
            pass
        
        # Fallback: assume offense (this shouldn't happen in normal training/inference)
        print("[WARNING] Could not extract role_flag from observation, defaulting to offense")
        batch_size = 1
        return torch.ones((batch_size, 1), dtype=torch.float32)

    
    def _get_action_logits(self, latent_pi: torch.Tensor) -> torch.Tensor:
        """
        Get action logits from the appropriate action network.
        
        If use_dual_policy is enabled, selects the appropriate action network
        based on the stored role_flags. Otherwise uses the standard action_net.
        
        This method is separated from _get_action_dist_from_latent to allow
        subclasses (like PassBiasDualCriticPolicy) to modify the logits before
        creating the distribution.
        
        :param latent_pi: Latent policy features
        :return: Action logits tensor
        """
        if self.use_dual_policy and self._current_role_flags is not None:
            # Use role-specific action networks
            role_flags = self._current_role_flags.to(latent_pi.device)
            
            # Get logits from both networks
            logits_offense = self.action_net_offense(latent_pi)
            logits_defense = self.action_net_defense(latent_pi)
            
            # Select appropriate logits based on role_flag
            # Handle shape: role_flags is (batch_size, 1), logits are (batch_size, action_dim)
            is_offense = (role_flags.squeeze(-1) > 0.0).unsqueeze(-1)  # (batch_size, 1)
            
            # Broadcast is_offense to match logits shape
            action_logits = torch.where(
                is_offense.expand_as(logits_offense),
                logits_offense,
                logits_defense
            )
            
            return action_logits
        else:
            # Use the standard single action network
            return self.action_net(latent_pi)
    
    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None
    ) -> Distribution:
        """
        Get action distribution from latent policy features.
        
        :param latent_pi: Latent policy features
        :param latent_sde: Latent SDE features (unused in this implementation)
        :return: Action distribution
        """
        action_logits = self._get_action_logits(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
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
        Evaluate actions according to the current policy, using the appropriate value network
        and action network.
        
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution.
        """
        # Extract role_flag BEFORE feature extraction
        role_flags = self._extract_role_flag(obs)
        
        # Store role_flags for use in _get_action_dist_from_latent
        self._current_role_flags = role_flags
        
        try:
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
        finally:
            # Clear stored role_flags
            self._current_role_flags = None

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.
        
        This override is necessary for dual policy support, as the base class's
        get_distribution doesn't extract role_flags before calling _get_action_dist_from_latent.
        
        :param obs: Observation
        :return: the action distribution.
        """
        # Extract role_flag BEFORE feature extraction
        role_flags = self._extract_role_flag(obs)
        
        # Store role_flags for use in _get_action_dist_from_latent
        self._current_role_flags = role_flags
        
        try:
            features = self.extract_features(obs)
            
            if self.share_features_extractor:
                latent_pi, _ = self.mlp_extractor(features)
            else:
                pi_features, _ = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
            
            return self._get_action_dist_from_latent(latent_pi)
        finally:
            # Clear stored role_flags
            self._current_role_flags = None
