"""SymbTr v3 txt dosyaları için ayrıştırıcı ve yardımcılar."""
from __future__ import annotations
import csv, io, os
from dataclasses import dataclass, field

TXT_DIR = os.path.join(os.path.dirname(__file__), "..", "symbtr_v3", "txt_v3")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "symbtr_v3", "pdf_v3")
NOTE_KOD = "9"      # asıl nota/sus satırları
REST = -1           # Koma53 == -1 -> sus


def read_text(path: str) -> str:
    """Dosyalar karışık kodlamalı: önce UTF-8, sonra Windows-1254 denenir."""
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp1254"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("cp1254", errors="replace")


def parse_name(fname: str) -> dict:
    """makam--form--usul--eser_adi--besteci. Bazı taksimlerde eser adı boş
    olduğundan alan sayısı 5'i aşabiliyor; ilk üç ve son alan sabit kabul edilir."""
    p = os.path.splitext(fname)[0].split("--")
    makam, form, usul = p[0], p[1], p[2].strip("-")
    besteci = p[-1].strip("-")
    eser = "--".join(p[3:-1]).strip("-") if len(p) > 4 else ""
    return dict(makam=makam, form=form, usul=usul, eser=eser, besteci=besteci)


@dataclass
class Piece:
    fname: str
    meta: dict
    rows: list = field(default_factory=list)   # tüm satırlar (dict)

    @property
    def notes(self):
        """Sesli notalar ve suslar: (koma53, pay, payda, ms, offset)."""
        out = []
        for r in self.rows:
            if r["Kod"] != NOTE_KOD:
                continue
            try:
                out.append((int(r["Koma53"]), int(r["Pay"]), int(r["Payda"]),
                            int(r["Ms"]), float(r["Offset"])))
            except ValueError:
                continue
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
