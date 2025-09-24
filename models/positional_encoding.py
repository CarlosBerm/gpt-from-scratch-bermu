import torch
import torch.nn as nn

class LearntPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # Create a learnable matrix [max_len x d_model]
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, word_embeddings):
        """
        Add learned positional embeddings to token embeddings.
        word_embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = word_embeddings.size()
        # positions = [0, 1, ..., seq_len-1]
        positions = torch.arange(seq_len, device=word_embeddings.device).unsqueeze(0)
        # positions: (1, seq_len)
        pos_embeddings = self.pe(positions)  # (1, seq_len, d_model)
        return word_embeddings + pos_embeddings


class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()

        # Positional encoding lookup table; matrix of 0s [max_len (r) x d_model (c)]
        pe = torch.zeros(max_len, d_model)

        # Sequence of numbers for each position that a token can have in the input
        pos = torch.arange(start=0, end=max_len, step=1, dtype=torch.float).unsqueeze(1)

        # Tensor to track the 'j' term 
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model)

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        """
        Use to add positional embeddings to the word embeddings"""
        # word_embeddings: (batch, seq_len, d_model)
        seq_len = word_embeddings.size(1)
        return word_embeddings + self.pe[:seq_len].unsqueeze(0)
