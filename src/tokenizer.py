"""Event-based tokenization of SymbTr pieces with four pitch encodings.

Sequence layout:  BOS [MAKAM:x USUL:y FORM:z]  ( [BAR] [POS:<n>/<d>] P:<pitch> D:<num>/<den> )*  EOS

POS (optional, `pos=True`) is the notated duration elapsed since the start of the current
measure, on an eighth-note grid (rounded down). It is a deterministic function of the
preceding tokens, so it gives the model an explicit running position inside the usul cycle
without adding information. The grid keeps the vocabulary small: exact positions would add
several hundred rare tokens.

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


POS_GRID, POS_MAX = 8, 127


def pos_token(f: Fraction) -> str:
    return f"POS:{min(int(f * POS_GRID), POS_MAX)}"


def note_tokens(events: list[dict], mode: str, karar: int | None, pos: bool = False) -> list[list[str]]:
    """One token group per note: [BAR]? [POS]? P D."""
    out, running = [], Fraction(0)
    for ev in events:
        g = []
        if is_bar_start(ev["onset"]):
            g.append(BAR); running = Fraction(0)
        if pos:
            g.append(pos_token(running))
        g += [pitch_token(ev, mode, karar), dur_token(ev)]
        running += Fraction(ev["num"], ev["den"])
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
        """'pitch', 'dur', 'pos', 'bar' or 'other' for a token id."""
        t = self.itos[i]
        return ("pitch" if t.startswith("P:") else "dur" if t.startswith("D:") else
                "pos" if t.startswith("POS:") else "bar" if t == BAR else "other")

    def fraction(self, i: int):
        """Fraction value of a D: token, else None."""
        t = self.itos[i]
        if t.startswith("D:"):
            n, d = t[2:].split("/"); return Fraction(int(n), int(d))
        return None


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
