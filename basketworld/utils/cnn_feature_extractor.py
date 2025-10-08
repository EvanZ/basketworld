"""
CNN + MLP Feature Extractor for BasketWorld
Combines spatial CNN processing with vector MLP features for hybrid learning.
"""
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import numpy as np


class HexCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor combining CNN for spatial features and MLP for vector features.
    
    Processes:
    - spatial: CNN for grid representation of the court (5 channels)
    - obs, skills, role_flag: MLP for vector features
    - action_mask: zeroed out (mask-agnostic, not used for learning)
    
    Architecture:
    - CNN branch: Conv layers → Flatten → Linear → ReLU
    - MLP branch: Concatenate vectors → Linear → ReLU → Linear → ReLU
    - Output: Concatenate both branches
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        cnn_features_dim: int = 256,
        mlp_features_dim: int = 128,
        cnn_channels: tuple = (32, 64, 64),
    ):
        """
        Args:
            observation_space: Dict observation space containing spatial, obs, skills, role_flag, action_mask
            cnn_features_dim: Output dimension of CNN branch
            mlp_features_dim: Output dimension of MLP branch
            cnn_channels: Tuple of CNN channel dimensions (e.g., (32, 64, 64))
        """
        # Calculate total output dimension
        features_dim = cnn_features_dim + mlp_features_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        # Extract subspaces
        spatial_space = observation_space.spaces.get("spatial")
        obs_space = observation_space.spaces["obs"]
        skills_space = observation_space.spaces["skills"]
        role_flag_space = observation_space.spaces["role_flag"]
        
        # CNN for spatial features (if spatial observation exists)
        if spatial_space is not None:
            n_input_channels = spatial_space.shape[0]
            
            cnn_layers = []
            in_channels = n_input_channels
            for out_channels in cnn_channels:
                cnn_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ])
                in_channels = out_channels
            
            self.cnn = nn.Sequential(*cnn_layers)
            
            # Calculate CNN output size dynamically
            with th.no_grad():
                sample_input = th.zeros(1, *spatial_space.shape)
                cnn_output = self.cnn(sample_input)
                cnn_flat_dim = int(np.prod(cnn_output.shape[1:]))
            
            self.cnn_linear = nn.Sequential(
                nn.Linear(cnn_flat_dim, cnn_features_dim),
                nn.ReLU(),
            )
        else:
            self.cnn = None
            self.cnn_linear = None
            cnn_features_dim = 0
        
        # MLP for vector features
        vector_dim = int(np.prod(obs_space.shape)) + \
                     int(np.prod(skills_space.shape)) + \
                     int(np.prod(role_flag_space.shape))
        
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, mlp_features_dim),
            nn.ReLU(),
            nn.Linear(mlp_features_dim, mlp_features_dim),
            nn.ReLU(),
        )
        
        self.cnn_enabled = self.cnn is not None
        self._features_dim = (cnn_features_dim if self.cnn_enabled else 0) + mlp_features_dim
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass combining CNN and MLP features.
        
        Args:
            observations: Dict containing spatial, obs, skills, role_flag, action_mask
            
        Returns:
            Concatenated feature tensor of shape (batch_size, features_dim)
        """
        features = []
        
        # Process spatial features with CNN
        if self.cnn_enabled and "spatial" in observations:
            spatial = observations["spatial"]
            cnn_out = self.cnn(spatial)
            cnn_out = cnn_out.reshape(cnn_out.size(0), -1)  # Flatten
            cnn_features = self.cnn_linear(cnn_out)
            features.append(cnn_features)
        
        # Process vector features with MLP
        # Zero out action_mask to prevent policy from learning directly from it
        vector_parts = [
            observations["obs"].reshape(observations["obs"].size(0), -1),
            observations["skills"].reshape(observations["skills"].size(0), -1),
            observations["role_flag"].reshape(observations["role_flag"].size(0), -1),
        ]
        vector_input = th.cat(vector_parts, dim=1)
        mlp_features = self.mlp(vector_input)
        features.append(mlp_features)
        
        # Concatenate all features
        return th.cat(features, dim=1)

