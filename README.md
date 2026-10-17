# mini-gpt

A small character-level GPT (decoder-only Transformer) written from scratch in PyTorch and trained on the Tiny Shakespeare dataset to generate Shakespeare-style text.

I built it while working through Andrej Karpathy's lecture [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), and the code is commented throughout with my own notes on what each step does and why.

## What the script does

[`02 Scripts/attention_model.py`](02%20Scripts/attention_model.py) runs end to end:

1. Downloads Tiny Shakespeare (~1.1 MB) into `01 Input/` on the first run
2. Builds a character-level tokenizer (65 unique characters)
3. Splits the text 90/10 into training and validation sets
4. Trains the model, printing training and validation loss every 250 steps
5. Generates 500 characters of new text from the trained model

## Model

| Setting | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Layers | 6 |
| Attention heads | 4 per layer (head size 16) |
| Embedding size | 64 |
| Context length | 64 characters |
| Dropout | 0.2 |
| Parameters | ~0.31M |
| Training | AdamW, learning rate 3e-4, batch size 32, 5,000 steps |

Token embeddings and learned position embeddings are added together, then passed through six blocks. Each block applies masked (causal) multi-head self-attention and then a feed-forward network (4x expansion, ReLU), each with LayerNorm beforehand and a residual connection around it. A final LayerNorm and linear layer produce logits for the next character.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python "02 Scripts/attention_model.py"
```

It uses a GPU if one is available and falls back to CPU otherwise. The model is small enough to train on a laptop CPU.

## Acknowledgements

Based on Andrej Karpathy's lecture above and his [nanoGPT](https://github.com/karpathy/nanoGPT) repository. Dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
