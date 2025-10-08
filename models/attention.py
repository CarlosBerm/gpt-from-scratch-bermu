import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.row_dim = 1
        self.col_dim = 2

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        """
        Apply the attention calculation, with an optional mask of booleans"""

         # encodings_*: (batch, seq_len, d_model)
        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)

        # (batch, seq_len, d_model) @ (batch, d_model, seq_len) -> (batch, seq_len, seq_len)
        simularity_score = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sim_score = simularity_score / q.size(self.col_dim)**0.5

        if mask is not None:
            scaled_sim_score = scaled_sim_score.masked_fill(mask, value=-1e9)
            
        attention_percents = F.softmax(scaled_sim_score, dim=self.col_dim)
        attention = torch.matmul(attention_percents, v)

        return attention


class MultiHeadAttention:
    """
    TODO work on in the future!"""
    def __init__(self, d_model):
        super().__init__()
        pass
