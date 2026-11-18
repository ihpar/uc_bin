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
    def notes(self):
        """Notes and rests as (koma53, numerator, denominator, ms, offset)."""
        out = []
        for r in self.rows:
            if r["Kod"] != NOTE_CODE:
                continue
            try:
                out.append((int(r["Koma53"]), int(r["Pay"]), int(r["Payda"]),
                            int(r["Ms"]), float(r["Offset"])))
            except ValueError:
                continue
        return out

    @property
    def events(self):
        """Notes and rests as dicts with onset/offset in measures:
        koma (Koma53, REST for rests), ae (AEU note name), num/den (notated
        duration as a fraction of a whole note), onset, offset."""
        out, prev = [], 0.0
        for r in self.rows:
            if r["Kod"] != NOTE_CODE:
                continue
            try:
                koma, num, den, off = int(r["Koma53"]), int(r["Pay"]), int(r["Payda"]), float(r["Offset"])
            except ValueError:
                continue
            if num <= 0 or den <= 0:
                continue
            out.append(dict(koma=koma, ae=r["NotaAE"], num=num, den=den, onset=prev, offset=off))
            prev = off
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
