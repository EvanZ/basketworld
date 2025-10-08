"""
Pure CNN Feature Extractor - encodes ALL features spatially (no MLP branch).
"""
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import numpy as np


class PureCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Pure CNN feature extractor that processes ONLY the spatial observation.
    
    This requires encoding ALL information spatially, including:
    - Shot clock (uniform channel)
    - Player skills (encoded in player positions)
    - Role flag (encoded as channel)
    
    For this to work, the environment must encode these in the spatial observation.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        features_dim: int = 512,
        cnn_channels: tuple = (32, 64, 128, 128),
    ):
        """
        Args:
            observation_space: Dict observation space (must contain 'spatial' key)
            features_dim: Output dimension
            cnn_channels: Tuple of CNN channel dimensions
        """
        super().__init__(observation_space, features_dim=features_dim)
        
        # Extract spatial subspace
        spatial_space = observation_space.spaces.get("spatial")
        if spatial_space is None:
            raise ValueError("PureCNNFeatureExtractor requires 'spatial' in observation space")
        
        n_input_channels = spatial_space.shape[0]
        
        # Build CNN layers
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
        
        # Final linear layer to desired output dimension
        self.linear = nn.Sequential(
            nn.Linear(cnn_flat_dim, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass - processes only spatial observation.
        
        Args:
            observations: Dict containing 'spatial' key
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        # Process spatial observation only
        spatial = observations["spatial"]
        cnn_out = self.cnn(spatial)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)  # Flatten
        features = self.linear(cnn_out)
        
        return features

