import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # On embedding: https://stackoverflow.com/questions/44881999/word-embedding-lookuptable-word-embedding-visualizations

    def forward(self, idx):
        """
        idx -> (B,T)
        targets -> (B,T)"""

        logits = self.token_embedding_table(idx) # (B,T,C)
        return logits

    def configure_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def training_step(self, input, target):
        logits = self.token_embedding_table(input)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = target.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] # (B, C); take last step

            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx