import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 250
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 30 # how many batches we use to estimate the final loss?
n_embd = 64
head_size = 16
n_heads = n_embd // head_size
n_layer = 6
drop_out = 0.2 # regularisation parameter (causes 20% of neurons to switch off each forward/backward pass)
                # It's like causing certain variables to be unsplittable in an RF

# ------------

torch.manual_seed(1337)

from pathlib import Path
import urllib.request

input_dir = Path(__file__).resolve().parent.parent / "01 Input"
input_file = input_dir / "input.txt"

# Make sure the folder exists
input_dir.mkdir(exist_ok=True)

# Download only if input.txt is not already there
if not input_file.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, input_file)
    print("Downloaded Shakespeare dataset.")
else:
    print("input.txt already exists — using existing file.")

# Read the text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset:", len(text))
print(text[:500])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters t o integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # this means that we are telling PyTorch no need to estimate gradients inside this function
                 # i.e., everything below we will not call .backwards on; this helps with memory and efficiency
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# SELF ATTENTION HEAD
class Head(nn.Module):
    ## Implements one head of self attention

    def __init__(self, head_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril((torch.ones(block_size, block_size))))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, targets=None):
        B, T, C = x.shape # during training this is generally batch_size, block_size but we extract them explicitly
                        # so that we can apply the same code during generation too                  
        q = self.query(x) # (B, T, H)
        k = self.key(x) # (B, T, H)
        v = self.value(x)
        wei =  q @ k.transpose(1, 2) * q.shape[-1]** (-0.5) # k.tranpose is now B, H, T; if we didn't have a batch dimension we'd just do: 
                                    # q @ k.T; because we have a batch dimension, we want to tell PyTorch transpose 
                                    # the T and H dimensions, but leave the batch dimension alone
                                    # wei becomes B, T, T because we have a T x H matrix times an H x T matrix
                                    # and conceptually we are after a matrix of key and query similarity scores 
                                    # (through dot products) which tells us, for each token in a block
                                    # how much information from preceeding tokens should be passed on to it
                                    # to update its hidden vector; T, T are called affinities; below we
                                    # make it a lower triangular matrix so that only preceeding tokens can provide
                                    # updates and softmax to produce normalised scores
                                    # essentially it's the matrix of the dot products between the key and query vectors
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # this says replace wei with -inf where tril equals 0
        wei = F.softmax(wei,dim =-1)  # softmax along the rows to get normalised scores; each row conssits of the 
                                    # attention scores of preceding tokens including itself, for each token        
        wei = self.dropout(wei)                
        out = wei @ v # this is still B, T, H so we will need to concatenate several (C divided by H lots) of these
                    # vectors to get something in the C dimension space
        return out

#Implement multi-headed attention
class MultiHeadAttention(nn.Module):
    # multiple heads of self-attention in parrallel
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # to project the concatenated vector back to the same space as something useful
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 

# Feed forward class - MLPs!
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.up = nn.Linear(n_embd, 4 * n_embd) # i think we have a bias for the up projection?
        self.down = nn.Linear(4 * n_embd, n_embd) # but don't for the down?
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = self.up(x)
        out = F.relu(out)
        out = self.down(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.feed = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.feed(self.ln2(x))
        return x
        

# Simple Biagram Language Model
class BiagramLanguageModel(nn.Module):
    ## Implements one head of self attention

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape # during training this is generally batch_size, block_size but we extract them explicitly
                        # so that we can apply the same code during generation too

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C); basically look up idx in the embedding table;
                                        # we get B lots of T x C matrices representing each chunk basically
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # T, C #why isn't it? self.position_embedding_table(idx)                                
        x = tok_emb + pos_emb
        x = self.blocks(x) # this takes x as an input and gives out; where are we adding this onto x?
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size); transform to be vocab dimensional with each hidden
                                        # vector being mapped from being embedding dimensional to vocab_size dimensional
                                        # giving a score for each potential next token    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last block_size
            idx_cond = idx[:, -block_size:] # B, T
            # get the predictions
            logits, loss = self(idx_cond) 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax along the rows to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BiagramLanguageModel()
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

m.eval()
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
