import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RevIN(nn.Module):

    """
        Reversible Instance Normalization pour la normalisation des séries temporelles
        from: https://openreview.net/pdf?id=cGDAkQo1C0p
        
        Permet de normalisé des série multivarié (à plusieurs features/canaux) de façon 
        individuel et stabilisé les valeurs => mean = 0 / std = 1.
        
        Cela permets de voir des pattern relatif et non absolue dans les séries
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
    
        """
            Args:
                * num_features (int):   nombre de canaux dans xs
                * eps          (float): pour la stabilité numérique (evite div par 0)
        """
        
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        self.stdev = 0
        self.mean  = 0
        
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
    
        """
            Up current statics about x (torch.Tensor)
        """
        
        dim2reduce = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else (1,)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach() # Move tensor from the GPU to CPU
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps # torch.var compute the variance
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