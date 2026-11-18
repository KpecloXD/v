import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, dropout=DROPOUT, batch_first=True)
        self.register_buffer("mask", torch.triu(torch.ones(BLOCK_SIZE, BLOCK_SIZE), diagonal=1).bool())

    def forward(self, x):
        length = x.size(1)
        out, _ = self.attn(x, x, x, attn_mask=self.mask[:length, :length], need_weights=False)
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class VLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > BLOCK_SIZE:
            raise ValueError("Последовательность длиннее BLOCK_SIZE")
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
