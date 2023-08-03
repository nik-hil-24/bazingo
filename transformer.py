# Imports
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from transformer_model import TransformerModel


# Seed
torch.manual_seed(1337)
# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Batch Size
BATCH_SIZE = 16
# Maximum Context Length
BLOCK_SIZE = 32
# Embedding Size
N_EMBED = 64
# Number of Layers of Transformer Block
N_LAYERS = 6
# Number of MHA Heads
NUM_HEADS = 6
# Dropout
DROPOUT = 0.0
# Printing Frequency
PRINT_FREQ = 500
# Epochs 
EPOCHS = 5000
# Learning Rate
LR = 1e-3


# Read Dataset
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
 
# Unique Characters in the Dataset
characters = list(set(text))
VOCAB_SIZE = len(characters)
print(''.join(sorted(characters)))
print(len(characters))

# Encode Text
mapping = {char:i for i, char in enumerate(characters)}
rev_mapping = dict(enumerate(characters))

encode_text = lambda string: [mapping[s] for s in string]
decode_text = lambda ls: ''.join([rev_mapping[l] for l in ls])

# Test
print(encode_text('Hi There!'))
print(decode_text(encode_text('Hi There!')))

# Creating Tensor Dataset of Encoded Text
data = torch.tensor(encode_text(text), dtype = torch.long)
print(data.shape)

# Train Test Split
n = int(0.9*(len(data)))
train = data[:n]
test = data[n:]


# DataLoader
def get_batch(split):
    # Get Data
    inp = train if split == 'train' else test
    # Random Indexes
    ix = torch.randint(len(inp)-BLOCK_SIZE, (BATCH_SIZE,))
    # x is from i:i+block_size, y is i+1:i_block_size+1
    x = torch.stack([inp[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([inp[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

x_batch, y_batch = get_batch('train')

# Model Evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    x, y = get_batch('val')
    x, y = x.to(device), y.to(device)
    _, loss = model(x, y)
    model.train()
    return loss


# Test Transformer Model
model = TransformerModel(BLOCK_SIZE, VOCAB_SIZE, N_EMBED, N_LAYERS, NUM_HEADS, DROPOUT, device).to(device)
logits, loss = model(x_batch.to(device), y_batch.to(device))
print(decode_text(model.generate(context = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))


# Optimizer
opt = AdamW(model.parameters(), lr = LR)


# Train Transformer Model
for epoch in range(EPOCHS):
    # Get Batch
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    # Forward
    logits, loss = model(x, y)

    # Backward
    opt.zero_grad(set_to_none = True)
    loss.backward()
    opt.step()

    if epoch%PRINT_FREQ == 0:
        val_loss = estimate_loss()
        print(f'Epoch: {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
        
context = torch.zeros((1, 1), dtype = torch.long).to(device)
print(decode_text(model.generate(context = context, max_new_tokens=500)[0].tolist()))