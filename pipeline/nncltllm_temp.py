import torch
import torch.nn as nn
import torch.nn.functional as F

# Composant du pipeline
from .modules.tsctp import TCTPLearner
from .modules.normalisation import RevIN
from .modules.patchembedding import PatchEmbedding

from transformers import GPT2Model, GPT2Config
from typing import List, Tuple, Optional

"""
	This core section of NNCLTLLM will be publish after some test 
	
	For now I made it unvailable
"""