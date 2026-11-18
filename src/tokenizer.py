"""Event-based tokenization of SymbTr pieces with four pitch encodings.

Sequence layout:  BOS [MAKAM:x USUL:y FORM:z]  ( [BAR] P:<pitch> D:<num>/<den> )*  EOS

Pitch encodings (`pitch_mode`):
  tet12       Koma53 rounded to the nearest 12-TET semitone (Western baseline, lossy)
  aeu         Arel-Ezgi-Uzdilek note name as written in SymbTr (NotaAE column)
  koma53      absolute 53-comma pitch (Koma53 column)
  koma53_rel  53-comma pitch relative to the karar (tonic) of the makam
"""
from __future__ import annotations
from fractions import Fraction
from .symbtr import REST

PITCH_MODES = ("tet12", "aeu", "koma53", "koma53_rel")
PAD, BOS, EOS, UNK, BAR = "<pad>", "<bos>", "<eos>", "<unk>", "BAR"
SPECIALS = [PAD, BOS, EOS, UNK, BAR]
_EPS = 1e-3


def is_bar_start(onset: float) -> bool:
    return abs(onset - round(onset)) < _EPS


def pitch_token(ev: dict, mode: str, karar: int | None) -> str:
    if ev["koma"] == REST:
        return "P:REST"
    if mode == "tet12":
        return f"P:{round(ev['koma'] * 12 / 53)}"
    if mode == "aeu":
        return f"P:{ev['ae']}"
    if mode == "koma53":
        return f"P:{ev['koma']}"
    if mode == "koma53_rel":
        return f"P:{ev['koma'] - karar}"
    raise ValueError(mode)


def dur_token(ev: dict) -> str:
    f = Fraction(ev["num"], ev["den"])
    return f"D:{f.numerator}/{f.denominator}"


def prefix_tokens(meta: dict) -> list[str]:
    return [f"MAKAM:{meta['makam']}", f"USUL:{meta['usul']}", f"FORM:{meta['form']}"]


def note_tokens(events: list[dict], mode: str, karar: int | None) -> list[list[str]]:
    """One token group per note: [BAR]? P D."""
    out = []
    for ev in events:
        g = [BAR] if is_bar_start(ev["onset"]) else []
        g += [pitch_token(ev, mode, karar), dur_token(ev)]
        out.append(g)
    return out


class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIALS + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk, self.bar = (self.stoi[t] for t in SPECIALS)

    def __len__(self): return len(self.itos)
    def encode(self, toks): return [self.stoi.get(t, self.unk) for t in toks]
    def decode(self, ids): return [self.itos[i] for i in ids]
    def kind(self, i: int) -> str:
        """'pitch', 'dur', 'bar' or 'other' for a token id."""
        t = self.itos[i]
        return "pitch" if t.startswith("P:") else "dur" if t.startswith("D:") else "bar" if t == BAR else "other"


def decode_pitch(tok: str, mode: str, karar: int | None, ae2koma: dict | None = None):
    """Token back to Koma53 (None if it cannot be decoded). tet12 is lossy by design."""
    v = tok[2:]
    if v == "REST": return REST
    try:
        if mode == "tet12": return round(int(v) * 53 / 12)
        if mode == "aeu": return ae2koma.get(v)
        if mode == "koma53": return int(v)
        if mode == "koma53_rel": return int(v) + karar
    except (ValueError, TypeError):
        return None
