"""Batched nucleus sampling with optional usul constraints.

Constraint (hard, applied at decoding time; works with any encoding):
  * a duration longer than what is left in the current measure is masked out,
  * BAR is forced exactly when the measure is full and forbidden otherwise,
  * a POS token, if the encoding uses them, is forced to the true running position.
The measure length comes from the usul (most frequent measure length in training).
"""
from __future__ import annotations
from fractions import Fraction
import torch, torch.nn.functional as F
from .tokenizer import pos_token


@torch.no_grad()
def generate_batch(model, heads: list[list[int]], vocab, max_new: int, stop_notes: int | None = None,
                   temperature: float = 1.0, top_p: float = 0.95,
                   measure_len: list | None = None, running: list | None = None,
                   progress: int = 0):
    """heads: prompt token ids per sequence. measure_len[i] is a Fraction (or None for no
    constraint); running[i] is the duration already elapsed in the current measure of the prompt.
    Returns the newly generated ids per sequence and whether each ended with EOS."""
    model.eval()
    dev = next(model.parameters()).device
    V = len(vocab); n = len(heads)
    kind = [vocab.kind(i) for i in range(V)]
    frac = [vocab.fraction(i) for i in range(V)]
    dur_ids = [i for i in range(V) if kind[i] == "dur"]
    has_pos = any(k == "pos" for k in kind)
    pos_id = lambda r: vocab.stoi.get(pos_token(r))
    seqs = [list(h) for h in heads]; new = [[] for _ in range(n)]
    done = [False] * n; eos = [False] * n; notes = [0] * n
    run = [Fraction(r) if r is not None else Fraction(0) for r in (running or [0] * n)]
    L = measure_len or [None] * n
    # what the previous token was: decides which constraint applies next
    last = [kind[h[-1]] if h and kind[h[-1]] in ("pitch", "dur", "pos", "bar") else "start" for h in heads]

    for step in range(max_new):
        act = [i for i in range(n) if not done[i]]
        if not act: break
        if progress and step and step % progress == 0:
            print(f"    step {step}/{max_new}, {len(act)}/{n} sequences still running", flush=True)
        T = min(max(len(seqs[i]) for i in act), model.ctx)
        X = torch.full((len(act), T), vocab.pad, dtype=torch.long)
        idx = []
        for r, i in enumerate(act):
            w = seqs[i][-model.ctx:]; X[r, :len(w)] = torch.tensor(w); idx.append(len(w) - 1)
        logits = model(X.to(dev)).float()[torch.arange(len(act)), torch.tensor(idx)].cpu() / temperature
        for r, i in enumerate(act):
            if L[i] is None: continue
            if last[i] in ("dur", "start"):                 # group boundary
                if run[i] >= L[i] or last[i] == "start" and run[i] == 0:
                    forced = vocab.bar
                else:
                    logits[r, vocab.bar] = float("-inf"); forced = None
                    if has_pos: forced = pos_id(run[i])
                if forced is not None:
                    m = torch.full((V,), float("-inf")); m[forced] = 0.0; logits[r] = m
            elif last[i] == "bar" and has_pos and pos_id(Fraction(0)) is not None:
                m = torch.full((V,), float("-inf")); m[pos_id(Fraction(0))] = 0.0; logits[r] = m
            elif last[i] == "pitch":                        # duration must fit the measure
                left = L[i] - run[i]
                ok = [d for d in dur_ids if frac[d] <= left]
                if ok:
                    m = torch.full((V,), float("-inf")); m[ok] = 0.0; logits[r] = logits[r] + m
        p = F.softmax(logits, -1)
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, -1) - sp < top_p
        sp = sp * keep
        pick = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(1).tolist()
        for r, i in enumerate(act):
            t = pick[r]
            if t == vocab.eos:
                done[i] = eos[i] = True; continue
            seqs[i].append(t); new[i].append(t)
            k = kind[t]
            if k == "bar": run[i] = Fraction(0)
            elif k == "dur":
                run[i] += frac[t]; notes[i] += 1
                if stop_notes and notes[i] >= stop_notes: done[i] = True
            if k in ("pitch", "dur", "pos", "bar"): last[i] = k
    return new, eos
