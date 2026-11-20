"""Builds token sequences for the baseline language models from data/splits.csv."""
from __future__ import annotations
import collections, csv, os, random
import torch
from .symbtr import load_piece, split_at_bar_lines, REST
from fractions import Fraction
from .tokenizer import Vocab, note_tokens, prefix_tokens, is_bar_start, BOS, EOS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_split_table(path: str | None = None) -> list[dict]:
    return list(csv.DictReader(open(path or os.path.join(ROOT, "data", "splits.csv"), encoding="utf-8")))


def load_corpus(splits=("train", "val", "test", "few_shot")) -> dict:
    """{split: [dict(fname, meta, events)]}"""
    out = collections.defaultdict(list)
    for row in load_split_table():
        if row["split"] not in splits:
            continue
        p = load_piece(row["fname"])
        ev = split_at_bar_lines(p.events)
        if len(ev) < 8:
            continue
        bars = [i for i, e in enumerate(ev) if is_bar_start(e["onset"])]
        lens = collections.Counter(sum(Fraction(q["num"], q["den"]) for q in ev[s:e]) for s, e in zip(bars, bars[1:]))
        out[row["split"]].append(dict(fname=row["fname"], meta=p.meta, events=ev, n_bars=len(bars),
                                      meter=lens.most_common(1)[0][0] if lens else None))   # notated measure length
    return out


def karar_by_makam(train: list[dict], extra: list[dict] = ()) -> dict:
    """Karar (tonic) per makam = most frequent final sounded pitch in the training set.
    For makams absent from training (the few-shot pool), the karar is estimated from the
    `extra` pieces: at inference time the karar of the chosen makam is known from theory,
    so this uses label knowledge only and no model training."""
    def count(pieces):
        c = collections.defaultdict(collections.Counter)
        for x in pieces:
            snd = [e["koma"] for e in x["events"] if e["koma"] != REST]
            if snd: c[x["meta"]["makam"]][snd[-1]] += 1
        return {m: v.most_common(1)[0][0] for m, v in c.items()}
    k = count(extra); k.update(count(train))
    return k


def measure_lengths(train: list[dict]) -> dict:
    """Usul -> most frequent notated measure length (fraction of a whole note) in training."""
    c = collections.defaultdict(collections.Counter)
    for x in train:
        ev = x["events"]
        bars = [i for i, e in enumerate(ev) if is_bar_start(e["onset"])]
        for s, e in zip(bars, bars[1:]):
            c[x["meta"]["usul"]][sum(Fraction(q["num"], q["den"]) for q in ev[s:e])] += 1
    return {u: v.most_common(1)[0][0] for u, v in c.items()}


def ae_to_koma(train: list[dict]) -> dict:
    """AEU note name -> most frequent Koma53 value (for decoding the 'aeu' encoding)."""
    c = collections.defaultdict(collections.Counter)
    for x in train:
        for e in x["events"]:
            if e["koma"] != REST: c[e["ae"]][e["koma"]] += 1
    return {a: v.most_common(1)[0][0] for a, v in c.items()}


def tokenize_corpus(corpus: dict, mode: str, karar: dict, cond: bool, pos: bool = False,
                    meter: bool = False, rem: bool = False, dur_first: bool = False):
    """Adds 'prefix' (token list) and 'groups' (list of per-note token lists) to every piece."""
    for pieces in corpus.values():
        for x in pieces:
            x["prefix"] = prefix_tokens(x["meta"], x["meter"] if meter else False) if cond else []
            x["groups"] = note_tokens(x["events"], mode, karar.get(x["meta"]["makam"]), pos, rem, dur_first)


def build_vocab(train: list[dict]) -> Vocab:
    toks = []
    for x in train:
        toks += x["prefix"]
        for g in x["groups"]: toks += g
    return Vocab(toks)


def encode_corpus(corpus: dict, vocab: Vocab):
    for pieces in corpus.values():
        for x in pieces:
            x["prefix_ids"] = vocab.encode(x["prefix"])
            ids, starts = [], []
            for g in x["groups"]:
                starts.append(len(ids)); ids += vocab.encode(g)
            x["ids"] = ids + [vocab.eos]
            x["note_starts"] = starts


class WindowSampler:
    """Random training windows: BOS + prefix + a slice of the piece starting on a note boundary.
    Targets for the prefix positions are masked so the model is not trained to guess the labels.

    The window position is drawn so that every token of a piece is covered equally often:
    windows may hang over either end of the piece and are then simply shorter. This matters
    for generation. Windows that begin at the first note teach the model how pieces open
    (composition from scratch), and endings appear at every position of the window, so the
    model cannot predict the end of a piece from the position alone."""

    MIN_TOKENS = 16

    def __init__(self, pieces: list[dict], vocab: Vocab, ctx: int, seed: int):
        self.pieces, self.vocab, self.ctx = pieces, vocab, ctx
        self.rng = random.Random(seed)
        self.weights = [len(x["ids"]) for x in pieces]     # sample pieces proportionally to length

    def batch(self, bs: int, device):
        X = torch.full((bs, self.ctx), self.vocab.pad, dtype=torch.long)
        Y = torch.full((bs, self.ctx), -100, dtype=torch.long)
        for b, x in enumerate(self.rng.choices(self.pieces, weights=self.weights, k=bs)):
            head = [self.vocab.bos] + x["prefix_ids"]
            room = self.ctx + 1 - len(head); n = len(x["ids"])
            s0 = self.rng.randint(self.MIN_TOKENS - room, max(n - self.MIN_TOKENS, 0))
            s = max((st for st in x["note_starts"] if st <= max(s0, 0)), default=0)
            e = min((s0 if s0 < 0 else s) + room, n)
            seq = head + x["ids"][s:e]
            m = len(seq) - 1
            X[b, :m] = torch.tensor(seq[:-1]); Y[b, :m] = torch.tensor(seq[1:])
            Y[b, :len(head) - 1] = -100
        return X.to(device), Y.to(device)


def eval_windows(x: dict, vocab: Vocab, ctx: int):
    """Deterministic windows covering a whole piece. Yields (input ids, target ids, score mask);
    windows overlap by half so that later tokens are scored with at least ctx/2 tokens of history."""
    head = [vocab.bos] + x["prefix_ids"]
    room = ctx + 1 - len(head); stride = max(room // 2, 1)
    ids, s, first = x["ids"], 0, True
    while True:
        seq = head + ids[s:s + room]
        tgt = seq[1:]
        mask = [False] * len(tgt)
        lo = len(head) - 1 + (0 if first else room - stride)
        for i in range(lo, len(tgt)): mask[i] = True
        yield seq[:-1], tgt, mask
        if s + room >= len(ids): break
        s += stride; first = False
