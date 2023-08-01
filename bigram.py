# Imports
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F


# Seed
torch.manual_seed(1337)
# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Batch Size
batch_size = 32
# Maximum Context Length
block_size = 8
# Embedding Size
n_embed = 32
# Printing Frequency
print_freq = 100
# Epochs 
epochs = 3000
# Learning Rate
lr = 1e-3


# Read Dataset
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
 
# Unique Characters in the Dataset
characters = list(set(text))
vocab_size = len(characters)
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
    ix = torch.randint(len(inp)-block_size, (batch_size,))
    # x is from i:i+block_size, y is i+1:i_block_size+1
    x = torch.stack([inp[i:i+block_size] for i in ix])
    y = torch.stack([inp[i+1:i+block_size+1] for i in ix])
    return x, y

x_batch, y_batch = get_batch('train')
for i in range(batch_size):
    for j in range(block_size):
        context = x_batch[i, :j+1]
        target = y_batch[i, j]
        print(f'Context is: {context.tolist()}, Target is {target.tolist()}')


# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed = 32):
        super(BigramLanguageModel, self).__init__()
        # Each Token Reads a Row From The Table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, context, targets = None):

        # Token embedding (batch_size, block_size, n_embed)
        token_embed = self.token_embedding_table(context)
        # Logits (batch_size, block_size, vocab_size)
        logits = self.lm_head(token_embed)

        # Loss
        if targets == None:
            loss = None
        else:
            # Get batch_size, block_size, vocab_size
            B, T, C = logits.shape
            # Reshape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T,)
            # Calculate Loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context, max_new_tokens):
        # Iterating Through Number of Tokens To Generate
        for _ in range(max_new_tokens):
            # Get Predictions
            logits, loss = self(context)
            # Get Last Block (Time Step)
            logits = logits[:, -1, :]
            # Probability
            probs = F.softmax(logits, dim = -1)
            # Sample From The Distribution
            context_next = torch.multinomial(probs, num_samples = 1)
            # Append
            context = torch.cat((context, context_next), dim = 1)

        return context
    

# Test Bigram
model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(x_batch.to(device), y_batch.to(device))
print(decode_text(model.generate(context = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))


# Optimizer
opt = AdamW(model.parameters(), lr = lr)


# Train BigramLanguageModel
for epoch in range(epochs):
    # Get Batch
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)

    # Forward
    logits, loss = model(x, y)

    # Backward
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if epoch%print_freq == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode_text(model.generate(context = context, max_new_tokens=50)[0].tolist()))
