import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RevIN(nn.Module):

    """
        Reversible Instance Normalization pour la normalisation des séries temporelles
        from: https://openreview.net/pdf?id=cGDAkQo1C0p
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: (B, T) ou (B, T, C), Batch, Timestamps, Channel or Feature
            mode: 'norm' pour normaliser, 'denorm' pour dénormaliser
        """
        
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
    
        dim2reduce = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else (1,)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
    
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self.stdev + self.mean
        return x

#CP