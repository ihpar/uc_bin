"""Small autoregressive language models: decoder-only Transformer and LSTM baseline."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class TransformerLM(nn.Module):
    def __init__(self, vocab: int, ctx: int, d_model=256, layers=6, heads=4, dropout=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight                 # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        T = x.size(1)
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.head(self.norm(self.enc(h, mask=mask)))


class LSTMLM(nn.Module):
    def __init__(self, vocab: int, ctx: int, d_model=256, layers=2, hidden=512, dropout=0.1, **_):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d_model)
        self.rnn = nn.LSTM(d_model, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        h, _ = self.rnn(self.drop(self.tok(x)))
        return self.head(self.drop(self.proj(h)))


def build_model(kind: str, vocab: int, cfg: dict) -> nn.Module:
    if kind == "transformer":
        return TransformerLM(vocab, cfg["ctx"], cfg["d_model"], cfg["layers"], cfg["heads"], cfg["dropout"])
    if kind == "lstm":
        return LSTMLM(vocab, cfg["ctx"], cfg["d_model"], cfg["lstm_layers"], cfg["lstm_hidden"], cfg["dropout"])
    raise ValueError(kind)


@torch.no_grad()
def generate(model, ids: list[int], max_new: int, eos: int, temperature=1.0, top_p=0.95, stop=None):
    """Nucleus sampling. `stop(ids)` may end generation early (e.g. after N notes)."""
    model.eval()
    dev = next(model.parameters()).device
    head_len = len(ids)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.ctx:]], device=dev)
        logits = model(x)[0, -1].float() / temperature
        p = F.softmax(logits, -1)
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, 0) - sp < top_p
        sp = sp * keep
        nxt = si[torch.multinomial(sp / sp.sum(), 1)].item()
        if nxt == eos: break
        ids = ids + [nxt]
        if stop and stop(ids[head_len:]): break
    return ids[head_len:]
