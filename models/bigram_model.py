import torch
import string
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx):
        """
        idx -> (B, T)
        targets -> (B, T)
        """

        logits = self.token_embedding_table(idx) # (B, T, C)
        return logits

    def training_step(self, logits):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

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
    
vocab_size = 65

model = BigramLanguageModel(vocab_size)
wordsList = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000).tolist()

alphabet = list(string.ascii_lowercase + string.ascii_uppercase)
letter_to_num = {letter: idx for idx, letter in enumerate(alphabet)}

num_to_letter = {idx: letter for letter, idx in letter_to_num.items()}

matched_letters = []
for num in wordsList[0]:
    letter = num_to_letter.get(num, '?')
    matched_letters.append(letter)

print("Generated letters:", matched_letters)