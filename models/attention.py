import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W_q =
        self.W_k =
        self.W_v =

        self.row_dim = 1
        self.col_dim = 2

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        """
        Apply the attention calculation, with an optional mask of booleans"""

         # encodings_*: (batch, seq_len, d_model)
        q = 
        k = 
        v = 

        # (batch, seq_len, d_model) @ (batch, d_model, seq_len) -> (batch, seq_len, seq_len)

        if mask is not None:
            # Mask logic
            
        attention_percents =

        return


class MultiHeadAttention:
    """
    TODO work on in the future!"""
    def __init__(self, d_model):
        super().__init__()
        pass
