"""Batched nucleus sampling with optional usul constraints.

Constraint (hard, applied at decoding time; works with any encoding):
  * a duration longer than what is left in the current measure is masked out,
  * BAR is forbidden inside a measure; when the measure is full the model may only start
    the next measure (BAR) or end the piece (EOS), so pieces end on a complete measure,
  * a POS token, if the encoding uses them, is forced to the true running position.
The measure length is the piece's meter if the model is conditioned on it, else the most
frequent measure length of the usul in training.

Countdown (for models trained with REM tokens; always on, it is part of their interface):
the REM token after each BAR is forced to the number of measures still to come, and once
the last measure has been written a further BAR is turned into the end of the piece.
"""
from __future__ import annotations
from fractions import Fraction
import torch, torch.nn.functional as F
from .tokenizer import pos_token, rem_token


@torch.no_grad()
def generate_batch(model, heads: list[list[int]], vocab, max_new: int, stop_notes: int | None = None,
                   temperature: float = 1.0, top_p: float = 0.95,
                   measure_len: list | None = None, running: list | None = None,
                   progress: int = 0, pinned: list | None = None, remaining: list | None = None):
    """heads: prompt token ids per sequence. measure_len[i] is a Fraction (or None for no
    constraint); running[i] is the duration already elapsed in the current measure of the prompt.
    pinned[i] is the number of leading tokens (BOS + conditioning prefix) that must stay
    visible when the sequence outgrows the context window, exactly as in training, where
    every window starts with BOS + prefix.
    remaining[i] is the REM value of the next measure to start (None = no countdown).
    Returns the newly generated ids per sequence and whether each ended with EOS."""
    model.eval()
    dev = next(model.parameters()).device
    V = len(vocab); n = len(heads)
    kind = [vocab.kind(i) for i in range(V)]
    frac = [vocab.fraction(i) for i in range(V)]
    dur_ids = [i for i in range(V) if kind[i] == "dur"]
    has_pos = any(k == "pos" for k in kind)
    pos_id = lambda r: vocab.stoi.get(pos_token(r))
    group_start = ("bar", "pos") if has_pos else ("bar", "pitch")
    seqs = [list(h) for h in heads]; new = [[] for _ in range(n)]
    done = [False] * n; eos = [False] * n; notes = [0] * n
    run = [Fraction(r) if r is not None else Fraction(0) for r in (running or [0] * n)]
    L = measure_len or [None] * n
    pin = pinned or [0] * n
    has_rem = any(k == "rem" for k in kind)
    nxt = list(remaining) if (remaining and has_rem) else [None] * n
    def only(row, ids):
        m = torch.full((V,), float("-inf")); m[ids] = 0.0; logits[row] = logits[row] + m
    # what the previous token was: decides which constraint applies next
    last = [kind[h[-1]] if h and kind[h[-1]] in ("pitch", "dur", "pos", "bar", "rem") else "start" for h in heads]

    for step in range(max_new):
        act = [i for i in range(n) if not done[i]]
        if not act: break
        if progress and step and step % progress == 0:
            print(f"    step {step}/{max_new}, {len(act)}/{n} sequences still running", flush=True)
        T = min(max(len(seqs[i]) for i in act), model.ctx)
        T = min(-(-T // 64) * 64, model.ctx)       # round up: fewer distinct tensor sizes on the GPU
        X = torch.full((len(act), T), vocab.pad, dtype=torch.long)
        idx = []
        for r, i in enumerate(act):
            w = seqs[i]
            if len(w) > model.ctx:                 # keep BOS + prefix, then the most recent tokens
                k = pin[i]; cut = len(w) - (model.ctx - k)
                for c in range(cut, min(cut + 8, len(w))):          # start on a note boundary, as in training
                    if kind[w[c]] in group_start:
                        cut = c; break                              # (keep the plain cut if none is near)
                w = w[:k] + w[cut:]
            X[r, :len(w)] = torch.tensor(w); idx.append(len(w) - 1)
        logits = model(X.to(dev)).float()[torch.arange(len(act)), torch.tensor(idx)].cpu() / temperature
        for r, i in enumerate(act):
            if nxt[i] is None: continue
            if last[i] == "bar":                            # REM token is determined by the countdown
                rid = vocab.stoi.get(rem_token(nxt[i] + 1))  # nxt was already decremented at the BAR
                if rid is not None: only(r, [rid])
            elif last[i] in ("dur", "start") and nxt[i] < 0:   # last measure written: a BAR means the end
                logits[r, vocab.eos] = torch.maximum(logits[r, vocab.eos], logits[r, vocab.bar])
                logits[r, vocab.bar] = float("-inf")
            elif last[i] in ("dur", "start"):
                logits[r, vocab.eos] = float("-inf")        # measures are still owed: no early ending
        for r, i in enumerate(act):
            if L[i] is None: continue
            if last[i] in ("dur", "start"):                 # group boundary
                allowed = None
                if last[i] == "start" and run[i] == 0:
                    allowed = [vocab.bar]                   # a piece starts with a bar line
                elif run[i] >= L[i]:                        # measure is full: next measure, or end the piece
                    allowed = [vocab.bar, vocab.eos] if nxt[i] is None else [vocab.eos] if nxt[i] < 0 else [vocab.bar]
                else:                                       # inside a measure: no bar line, no ending
                    logits[r, vocab.bar] = float("-inf"); logits[r, vocab.eos] = float("-inf")
                    if has_pos and pos_id(run[i]) is not None: allowed = [pos_id(run[i])]
                if allowed is not None:
                    m = torch.full((V,), float("-inf")); m[allowed] = 0.0; logits[r] = logits[r] + m
            elif last[i] in ("bar", "rem") and has_pos and pos_id(Fraction(0)) is not None \
                    and not (last[i] == "bar" and nxt[i] is not None):
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
            if k == "bar":
                run[i] = Fraction(0)
                if nxt[i] is not None: nxt[i] -= 1
            elif k == "dur":
                run[i] += frac[t]; notes[i] += 1
                if stop_notes and notes[i] >= stop_notes: done[i] = True
            if k in ("pitch", "dur", "pos", "bar", "rem"): last[i] = k
    return new, eos
