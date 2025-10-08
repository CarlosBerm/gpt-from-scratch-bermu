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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism that applies multiple attention heads in parallel
    and concatenates their outputs.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension of each head
        
        # Linear projections for all heads combined
        # Projecting all the heads together (e.g. one big matrix for everyone combined)
        # Is the same size because we're splitting the model into num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection 
        # The big weight output matrix (the value up matrix) is also concatted together to the same size)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.row_dim = 2
        self.col_dim = 3

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention for multiple heads
        
        Args:
            q, k, v: (batch, num_heads, seq_len, d_k)
            mask: (batch, seq_len, seq_len) or broadcastable
        
        Returns:
            attention: (batch, num_heads, seq_len, d_k)
        """
        # Compute attention scores
        # (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        similarity_score = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        scaled_sim_score = similarity_score / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scaled_sim_score = scaled_sim_score.masked_fill(mask, value=-1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scaled_sim_score, dim=-1)
        
        # Apply attention to values
        attention = torch.matmul(attention_weights, v)
        
        return attention
    
    def forward(self, encodings_q, encodings_k, encodings_v, mask=None):
        """
        Apply multi-head attention calculation, with an optional mask of booleans
        
        Args:
            encodings_q: (batch, seq_len, d_model)
            encodings_k: (batch, seq_len, d_model)
            encodings_v: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or broadcastable
        
        Returns:
            attention: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = encodings_q.size()
        
        # Apply linear transformations and split into multiple heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        q = self.W_q(encodings_q).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.W_k(encodings_k).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.W_v(encodings_v).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to (batch, num_heads, seq_len, d_k)
        # Such that the scaled dot prod can be applied across grouped by heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention = self._scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Apply output projection
        output = self.W_o(attention)
        
        return output
    
    


