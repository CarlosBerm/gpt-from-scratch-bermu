import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pass

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        """
        Apply the attention calculation, with an optional mask of booleans"""

        # TODO: Implement


class MultiHeadAttention:
    """
    TODO work on in the future!"""
    def __init__(self, d_model):
        super().__init__()
        pass
