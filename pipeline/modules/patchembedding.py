import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """Convertit les séries temporelles en patches avec embeddings"""
    
    def __init__(
        self, 
        patch_len: int = 16, 
        stride: int = 8, 
        embedding_dim: int = 768,
        input_channels: int = 1
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        
        # Conv1D pour extraire les patches (channel-independent)
        self.patch_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=embedding_dim,
            kernel_size=patch_len,
            stride=stride
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) séries temporelles multivariées
        Returns:
            patches: (B, N, D) où N est le nombre de patches
        """
        # x shape: (B, C, T)
        patches = self.patch_conv(x)  # (B, D, N)
        patches = patches.transpose(1, 2)  # (B, N, D)
        
        return patches
