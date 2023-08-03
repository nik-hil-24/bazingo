import torch
from torch import nn
from torch.nn import functional as F


# Self Attention
class Head(nn.Module):
    """One Head of Self Attention"""
    def __init__(self, block_size, n_embed, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get Input Shape
        B, T, C = x.shape

        # Key, Query, Values
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Masked Attention
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        out = wei@v


        return out


# Multi-Head Attention
class MHA(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, block_size, n_embed, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.projection(x))
        
        return out


# FeedForward Network
class FeedForward(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block
class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, block_size, n_embed, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MHA(block_size, n_embed, num_heads, head_size, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        out = x + self.ff(self.ln2(x))
        
        return out
    

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embed, n_layers, num_heads, dropout, device):
        super(TransformerModel, self).__init__()
        # Model Variables
        self.device = device
        self.block_size = block_size
        
        # Each Token Reads a Row From The Table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Token Position Embedding Table
        self.pos_embed_table = nn.Embedding(n_embed, n_embed)
        # Transformer Table
        self.blocks = nn.Sequential(*[Block(block_size, n_embed, num_heads, dropout) for _ in range(n_layers)])
        # Layer Norm
        self.ln = nn.LayerNorm(n_embed)
        # Final Layer
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, context, targets = None):
        # Get Shape
        B, T = context.shape
        
        # Token embedding (batch_size, block_size, n_embed)
        token_embed = self.token_embedding_table(context)
        # Positional Embedding
        pos_embed = self.pos_embed_table(torch.arange(T, device = self.device))
        # Adding Positional Embedding
        x = token_embed + pos_embed
        # Transformer Block
        x = self.blocks(x)
        # Logits (batch_size, block_size, vocab_size)
        logits = self.lm_head(self.ln(x))

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
        for i in range(max_new_tokens):
            # crop context to the last block_size tokens
            context_new = context[:, -self.block_size:]
            # Get Predictions
            logits, loss = self(context_new)
            # Get Last Block (Time Step)
            logits = logits[:, -1, :]
            # Probability
            probs = F.softmax(logits, dim = -1)
            # Sample From The Distribution
            context_next = torch.multinomial(probs, num_samples = 1)
            # Append
            context = torch.cat((context, context_next), dim = 1)

        return context
    
if __name__ == '__main__':
    # Params
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    block_size = 8
    vocab_size = 64
    n_embed = 384
    n_layers = 6
    num_heads = 6
    dropout = 0.2

    # Model
    model = TransformerModel(block_size, vocab_size, n_embed, n_layers, num_heads, dropout, device).to(device)
    
    # Generate
    context = torch.zeros((1, 1), dtype = torch.long, device = device)
    print(model.generate(context, max_new_tokens = 500)[0].tolist())