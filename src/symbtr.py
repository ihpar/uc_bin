"""Parser and helpers for SymbTr v3 txt files."""
from __future__ import annotations
import csv, io, os
from dataclasses import dataclass, field

_ROOT = os.path.join(os.path.dirname(__file__), "..", "symbtr_v3")
TXT_DIR = os.path.join(_ROOT, "txt_v3")
PDF_DIR = os.path.join(_ROOT, "pdf_v3")
MU2_DIR = os.path.join(_ROOT, "mu2_v3")
XML_DIR = os.path.join(_ROOT, "xml_v3")
NOTE_CODE = "9"     # value of the "Kod" column for regular note/rest rows
REST = -1           # Koma53 == -1 marks a rest


def read_text(path: str) -> str:
    """Files use mixed encodings: try UTF-8 first, then Windows-1254."""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp1254"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("cp1254", errors="replace")


def parse_name(fname: str) -> dict:
    """File names follow makam--form--usul--title--composer. Some taksim files
    have an empty title, which yields more than five fields; the first three
    fields and the last one are treated as fixed."""
    p = os.path.splitext(fname)[0].split("--")
    makam, form, usul = p[0], p[1], p[2].strip("-")
    composer = p[-1].strip("-")
    title = "--".join(p[3:-1]).strip("-") if len(p) > 4 else ""
    return dict(makam=makam, form=form, usul=usul, title=title, composer=composer)


@dataclass
class Piece:
    fname: str
    meta: dict
    rows: list = field(default_factory=list)   # all rows as dicts (original column names)

    @property
    def events(self):
        """Every row that occupies time, as dicts with onset/offset in measures:
        koma (Koma53, REST for rests), ae (AEU note name), num/den (notated duration as a
        fraction of a whole note), onset, offset, kod (SymbTr row code).

        A row occupies time when its Offset is later than that of the previous row. Besides
        regular notes and rests (code 9) this covers notes that carry an ornament or
        articulation (tremolo, trill, mordent, glissando, notes with a grace note, small
        notes, ...), which SymbTr stores under their own codes, and the few rests stored with
        a tempo change. Grace notes without a duration of their own (code 8) and pure
        markers (usul/tempo changes, layout codes) are skipped. See the SymbTr code table."""
        out, prev = [], 0.0
        for r in self.rows:
            try:
                num, den, off = int(r["Pay"]), int(r["Payda"]), float(r["Offset"])
            except ValueError:
                continue
            if off - prev <= 1e-6:
                continue                      # marker or zero-length grace note
            if num > 0 and den > 0:
                try: koma = int(r["Koma53"])
                except ValueError: koma = REST
                out.append(dict(koma=koma if koma > 0 else REST, ae=r["NotaAE"], num=num, den=den,
                                onset=prev, offset=off, kod=r["Kod"]))
            prev = off
        return out

    @property
    def notes(self):
        """Time-occupying rows as (koma53, numerator, denominator, ms, offset) tuples."""
        return [(e["koma"], e["num"], e["den"], 0, e["offset"]) for e in self.events]


def split_at_bar_lines(events: list) -> list:
    """Split every note or rest that is held across a bar line into tied parts, as a printed
    score does. The txt files store such a note once, with an Offset beyond the bar line;
    without splitting, the measure would have no start and its durations would not add up.
    Continuation parts get tie=True."""
    from fractions import Fraction
    import math
    out = []
    for e in events:
        a, b = e["onset"], e["offset"]
        cuts = [m for m in range(math.floor(a) + 1, math.ceil(b)) if m - a > 0.01 and b - m > 0.01]
        if not cuts:
            out.append(dict(e, tie=False)); continue
        total = Fraction(e["num"], e["den"]); edges = [a] + cuts + [b]
        for i, (x, y) in enumerate(zip(edges, edges[1:])):
            part = total * Fraction((y - x) / (b - a)).limit_denominator(64)
            out.append(dict(e, onset=float(x), offset=float(y), num=part.numerator, den=part.denominator,
                            tie=i > 0 and e["koma"] != REST))
    return out


def load_piece(fname: str, txt_dir: str = TXT_DIR) -> Piece:
    text = read_text(os.path.join(txt_dir, fname))
    rd = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = [r for r in rd if r.get("Kod")]
    return Piece(fname, parse_name(fname), rows)


def iter_pieces(txt_dir: str = TXT_DIR):
    for f in sorted(os.listdir(txt_dir)):
        if f.endswith(".txt"):
            yield load_piece(f, txt_dir)
