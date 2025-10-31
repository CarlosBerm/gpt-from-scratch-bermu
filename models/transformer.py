import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import LearntPositionEncoding
from attention import MultiHeadAttention

def create_causal_mask(seq_len: int, device: torch.device):
    """Upper-triangular (excluding diagonal) True=masked causal matrix of shape (1, S, S)."""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
         
         super().__init__()

         # define the 2 normalization layers
         self.layerNorm1 = nn.LayerNorm(d_model)
         self.layerNorm2 = nn.LayerNorm(d_model)

         self.attention = MultiHeadAttention(d_model, num_heads)

         # define the sequential feed-forward layer, consisting of:
         # 1. Linear layer, 2. Relu layer, 3. Linear layer
         self.ff = nn.Sequential(
              nn.Linear(d_model, d_ff),
              nn.GELU(),
              nn.Linear(d_ff, d_model),
         )

         self.drop1 = nn.Dropout(dropout) # Dropouts are OPTIONAL!!!
         self.drop2 = nn.Dropout(dropout)
          
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
          # pass thru multi-head attention
          a = self.attention(self.layerNorm1(x), self.layerNorm1(x), self.layerNorm1(x), mask=mask)
          x = x + self.drop1(a)

          f = self.ff(self.layerNorm2(x))
          x = x + self.drop2(f)

          return x


class DecoderOnlyTransformer(nn.Module):
     def __init__(self, vocab_size, d_model, num_heads, d_ff, n_layers, max_len):
          super().__init__()
          
          # define the rest of the constructor parameters
          self.vocab_size = vocab_size
          self.d_model = d_model
          self.num_heads = num_heads
          self.d_ff = d_ff
          self.n_layers = n_layers
          self.max_len = max_len

          #define the token and positional embeddings
          self.tok_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
          self.pos_embed = LearntPositionEncoding(d_model=d_model, max_len=max_len)
          self.drop = nn.Dropout(p=0.1)

          #define the transformer blocks
          self.blocks = nn.ModuleList([
               TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.1)
               for _ in range(n_layers)
          ])

          #define the final layer norm (NOT THE SANE AS THE ONES WITHIN THE TRANSFORMER BLOCK)
          self.final_layer_norm = nn.LayerNorm(d_model)

          #define the final output projection using self.head
          self.head = nn.Linear(in_features=d_model, out_features=vocab_size, bias=False)
          # weight tying with token embeddings (GPT-2 style)
          self.head.weight = self.tok_embed.weight

          # convenience loss (can be overridden)
          self.loss = nn.CrossEntropyLoss()

     
     def forward(self, token_ids):
          # 12 blocks in total
          #define batch size, seq_len
          bsz, seq_len = token_ids.shape

          #get token embeddings
          x = self.tok_embed(token_ids)
          #get positional embeddings
          x = self.pos_embed(x)
          x = self.drop(x)

          #create mask
          causal = create_causal_mask(seq_len, token_ids.device).expand(bsz, -1, -1)

          #pass through transformer blocks
          for blk in self.blocks:
               x = blk(x, mask=causal)

          #pass through final layer norm and output
          x = self.final_layer_norm(x)
          logits = self.head(x)
          return logits

     
     def training_step(self, input, target):
          #hint: very similar to bigram
          logits = self.forward(input)
          b, s, v = logits.shape
          loss = self.loss(logits.view(b*s, v), target.view(b*s))
          return loss
     
     def generate(self, idx, max_new_tokens, temperature: float = 1.0, top_k: int | None = None):
          #hint: very similar to bigram
          for _ in range(max_new_tokens):
               # crop to max context if needed
               if idx.size(1) > self.max_len:
                    idx = idx[:, -self.max_len:]
               logits = self.forward(idx)
               logits = logits[:, -1, :] / max(temperature, 1e-8)

               if top_k is not None and 0 < top_k < logits.size(-1):
                    values, _ = torch.topk(logits, k=top_k, dim=-1)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)

               probs = F.softmax(logits, dim=-1)
               next_token = torch.multinomial(probs, num_samples=1)
               idx = torch.cat([idx, next_token], dim=1)
          
          return idx
