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
        Apply the attention calculation, with an optional mask of booleans
        """
        # encodings_*: (batch, seq_len, d_model)
        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)

        # (batch, seq_len, d_model) @ (batch, d_model, seq_len) -> (batch, seq_len, seq_len)
        similarityScore = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaledSimScore = similarityScore / q.size(self.col_dim) ** 0.5

        if mask is not None:
            # Mask logic
            scaledSimScore = scaledSimScore.masked_fill(mask, value=float('-inf'))

        attention_percents = F.softmax(scaledSimScore, dim=self.col_dim)
        attention = torch.matmul(attention_percents, v)

        return attention


class MultiHeadAttention:
    """
    work on in the future!
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # dimension of each head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.row_dim = 2
        self.col_dim = 3
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        q, k, v: (batch, num_heads, seq_len, d_k)
        """
        similarityScore = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        scaledSimScore = similarityScore / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scaledSimScore = scaledSimScore.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scaledSimScore, dim=self.col_dim)

        attention = torch.matmul(attention_weights, v)  # (batch, num_heads, seq_len, d_k)

        return attention

    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        batch_size, seq_len, _ = encodings_q.size()

        q = self.W_q(encodings_q).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.W_k(encodings_k).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.W_v(encodings_v).view(batch_size, seq_len, self.num_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention = self.scaled_dot_product_attention(q, k, v, mask)  # (batch, num_heads, seq_len, d_k)

        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_o(attention)  # (batch, seq_len, d_model)

        return output
    