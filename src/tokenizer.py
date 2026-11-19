"""Event-based tokenization of SymbTr pieces with four pitch encodings.

Sequence layout:
  BOS [MAKAM:x USUL:y FORM:z [METER:n/d]]  ( [BAR [REM:k]] [POS:k] [TIE] P:<pitch> D:<num>/<den> )*  EOS

TIE marks the continuation of a note held across a bar line (the note is split there, as in
a printed score).

METER (optional) is the notated measure length of the piece. The same usul is written with
different measure lengths in the collection (e.g. duyek as 4/4 or 8/4), so the usul name
alone does not fix it.

REM (optional, `rem=True`) follows every BAR and counts the measures that remain after the
one that starts there (REM:0 = last measure; values above REM_MAX are written REM:many).
It tells the model how far it is from the end of the piece, and lets the user of the editor
choose the length of a composition.

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
PAD, BOS, EOS, UNK, BAR, TIE = "<pad>", "<bos>", "<eos>", "<unk>", "BAR", "TIE"
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


def prefix_tokens(meta: dict, meter=False) -> list[str]:
    """`meter`: False = no METER token; otherwise a Fraction, or None for unmetered pieces."""
    out = [f"MAKAM:{meta['makam']}", f"USUL:{meta['usul']}", f"FORM:{meta['form']}"]
    return out if meter is False else out + [meter_token(meter)]


POS_GRID, POS_MAX = 8, 127
REM_MAX = 16


def rem_token(k: int) -> str:
    return f"REM:{k}" if k <= REM_MAX else "REM:many"


def meter_token(meter) -> str:
    return f"METER:{meter.numerator}/{meter.denominator}" if meter else "METER:free"


def pos_token(f: Fraction) -> str:
    return f"POS:{min(int(f * POS_GRID), POS_MAX)}"


def note_tokens(events: list[dict], mode: str, karar: int | None, pos: bool = False,
                rem: bool = False) -> list[list[str]]:
    """One token group per note: [BAR [REM]]? [POS]? [TIE]? P D."""
    out, running = [], Fraction(0)
    left = sum(is_bar_start(ev["onset"]) for ev in events)      # measures not yet started
    for ev in events:
        g = []
        if is_bar_start(ev["onset"]):
            g.append(BAR); running = Fraction(0); left -= 1
            if rem: g.append(rem_token(left))
        if pos:
            g.append(pos_token(running))
        if ev.get("tie"): g.append(TIE)
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
        """'pitch', 'dur', 'pos', 'rem', 'bar' or 'other' for a token id."""
        t = self.itos[i]
        return ("pitch" if t.startswith("P:") else "dur" if t.startswith("D:") else
                "pos" if t.startswith("POS:") else "rem" if t.startswith("REM:") else
                "bar" if t == BAR else "tie" if t == TIE else "other")

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
