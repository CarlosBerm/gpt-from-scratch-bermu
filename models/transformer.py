import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import PositionEncoding
from attention import MultiHeadAttention

def _create_causal_mask(self, seq_len, device):
     mask = torch.triu(torch.ones(seq_len, seq_len, device=device), daigonal = 1).bool()
     return mask

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads=12):
         self.attention = MultiHeadAttention(d_model, num_heads)
         #define the 2 normalization layers
         #define the sequential feed-forward layer, consisting of:
         # 1. Linear layer, 2. Relu layer, 3. Linear layer
    
    def forward(self, x, mask=None):
         
          # TODO:
          # multi-head attention -> 12 attention heads per block

          #pass through attention layer
          #pass through norm1 layer
          #pass through feed-forward layer
          #pass through norm2 layer

          return x


class DecoderOnlyTransformer(nn.Module):
     def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers,
                 max_len):
          
          self.vocab_size = vocab_size
          #define the rest of the constructor parameters
          #define the token and positional embeddings
          #define the transformer blocks
          #define the final layer norm (NOT THE SANE AS THE ONES WITHIN THE TRANSFORMER BLOCK)
          #define the final output projection using self.head

     
          
     
     def forward(self, token_ids):
          # 12 blocks in total
          #define batch size, seq_len

          #get token embeddings
          #get positional embeddings

          #create mask
          #pass through transformer blocks

          #pass through final layer norm and output
          pass

     
     def training_step(self, input, target):
          #hint: very similar to bigram

          return 
     
     def generate(self, idx, max_new_tokens):
          #hint: very similar to bigram
          for _ in range(max_new_tokens):
               pass
          

          return idx

          
