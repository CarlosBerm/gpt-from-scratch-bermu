
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple
import requests
import os

from models.bigram_model import BigramLanguageModel

# <=====- Tokenizer -======>
# tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
# tokenizer.pad_token = tokenizer.eos_tokenv


# <=====- Dataset -======>
# ds = load_dataset("PleIAs/common_corpus", split="train[:0.5%]") 

text = ""
if not os.path.isfile("./shakespeare.txt"):
    url = "https://www.gutenberg.org/files/100/100-0.txt"  # Complete Works of Shakespeare
    response = requests.get(url)
    text = response.text

    with open("shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(text)
        print("Saved Shakespeare works to shakespeare.txt")
else:
    print("Found Shakespeare works in directory")
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
       text = f.read()

print("Tokenizing")
# ids = tokenizer(text[:500])["input_ids"]
# data = torch.tensor(ids, dtype=torch.long)
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
vocab_size = len(chars)
print("Finished Tokenizing")


n = int(0.9*len(data)) # first 90% will be train, rest val
print("split:", n)
train_data = data[:n]
val_data = data[n:]

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def train(model, epochs = 1, batch_size = 256, device = None):
    print("Beginning training")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    optimizer = model.configure_optimizer()
    model.train()

    batches = epochs
    total_loss = 0
    for i in range(batches):
        xb, yb = get_batch(train_data)
        xb, yb = xb.to(device), yb.to(device)
        loss = model.training_step(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach())

        avg_loss = total_loss / max(1, len(data))
        print(f"Step {i+1}/{epochs} - Loss: {avg_loss:.4f}")

        # save the model
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size)
    
    train(model, epochs=100000)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    print("Generating...")
    generated_ids = model.generate(context, max_new_tokens=1000)[0].tolist()
    print(decode(generated_ids))
    with open("generated.txt", "w", encoding="utf-8") as f:
        f.write(decode(generated_ids))
        print("Saved generated text to generated.txt")
