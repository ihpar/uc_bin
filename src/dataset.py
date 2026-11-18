"""Builds token sequences for the baseline language models from data/splits.csv."""
from __future__ import annotations
import collections, csv, os, random
import torch
from .symbtr import load_piece, REST
from .tokenizer import Vocab, note_tokens, prefix_tokens, BOS, EOS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_split_table(path: str | None = None) -> list[dict]:
    return list(csv.DictReader(open(path or os.path.join(ROOT, "data", "splits.csv"), encoding="utf-8")))


def load_corpus(splits=("train", "val", "test")) -> dict:
    """{split: [dict(fname, meta, events)]}"""
    out = collections.defaultdict(list)
    for row in load_split_table():
        if row["split"] not in splits:
            continue
        p = load_piece(row["fname"])
        ev = p.events
        if len(ev) < 8:
            continue
        out[row["split"]].append(dict(fname=row["fname"], meta=p.meta, events=ev))
    return out


def karar_by_makam(train: list[dict]) -> dict:
    """Karar (tonic) per makam = most frequent final sounded pitch in the training set."""
    c = collections.defaultdict(collections.Counter)
    for x in train:
        snd = [e["koma"] for e in x["events"] if e["koma"] != REST]
        if snd: c[x["meta"]["makam"]][snd[-1]] += 1
    return {m: v.most_common(1)[0][0] for m, v in c.items()}


def ae_to_koma(train: list[dict]) -> dict:
    """AEU note name -> most frequent Koma53 value (for decoding the 'aeu' encoding)."""
    c = collections.defaultdict(collections.Counter)
    for x in train:
        for e in x["events"]:
            if e["koma"] != REST: c[e["ae"]][e["koma"]] += 1
    return {a: v.most_common(1)[0][0] for a, v in c.items()}


def tokenize_corpus(corpus: dict, mode: str, karar: dict, cond: bool):
    """Adds 'prefix' (token list) and 'groups' (list of per-note token lists) to every piece."""
    for pieces in corpus.values():
        for x in pieces:
            x["prefix"] = prefix_tokens(x["meta"]) if cond else []
            x["groups"] = note_tokens(x["events"], mode, karar.get(x["meta"]["makam"]))


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
    Targets for the prefix positions are masked so the model is not trained to guess the labels."""

    def __init__(self, pieces: list[dict], vocab: Vocab, ctx: int, seed: int):
        self.pieces, self.vocab, self.ctx = pieces, vocab, ctx
        self.rng = random.Random(seed)
        self.weights = [len(x["ids"]) for x in pieces]     # sample pieces proportionally to length

    def batch(self, bs: int, device):
        X = torch.full((bs, self.ctx), self.vocab.pad, dtype=torch.long)
        Y = torch.full((bs, self.ctx), -100, dtype=torch.long)
        for b, x in enumerate(self.rng.choices(self.pieces, weights=self.weights, k=bs)):
            head = [self.vocab.bos] + x["prefix_ids"]
            room = self.ctx + 1 - len(head)
            s = self.rng.choice(x["note_starts"]) if len(x["ids"]) > room else 0
            if len(x["ids"]) - s < room and len(x["ids"]) > room:
                s = max(st for st in x["note_starts"] if st <= len(x["ids"]) - room)
            seq = head + x["ids"][s:s + room]
            n = len(seq) - 1
            X[b, :n] = torch.tensor(seq[:-1]); Y[b, :n] = torch.tensor(seq[1:])
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
