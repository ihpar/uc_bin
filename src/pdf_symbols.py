"""Automatic symbol bounding boxes from SymbTr PDFs (vector, embedded Mus2 font).

Every music symbol is a character of the Mus2 font. The character position on
the page comes from the PDF; the tight bounding box comes from the glyph bounds
of the embedded font. Staff lines, barlines, beams, slurs etc. are vector
drawings rather than font characters and are returned by `page_drawings`."""
from __future__ import annotations
import io, logging
import pymupdf
from fontTools.ttLib import TTFont

logging.getLogger("fontTools").setLevel(logging.ERROR)
MUSIC_FONT = "Mus2"


def _glyph_tables(doc, page) -> dict:
    """{font name: (cmap, glyf, unitsPerEm)} for the music fonts on the page."""
    out = {}
    for f in page.get_fonts(full=True):
        if MUSIC_FONT not in f[3]:
            continue
        if f[1] != "ttf" or f[2] != "TrueType":
            continue                # CFF/Type0 embeddings: see font_kind()
        buf = doc.extract_font(f[0])[3]
        if not buf:
            continue
        try:
            t = TTFont(io.BytesIO(buf))
        except Exception:
            continue
        cmap = {}
        for tb in t["cmap"].tables:
            cmap.update(tb.cmap)
        out[f[3]] = (cmap, t["glyf"], t["head"].unitsPerEm)
    return out


def _code(ch: str):
    """Map the Unicode character reported by MuPDF back to the font byte code."""
    try:
        return ch.encode("cp1252")[0]
    except (UnicodeEncodeError, IndexError):
        return ord(ch) if ord(ch) < 256 else None


def page_symbols(doc, page) -> list[dict]:
    """Return [{code, bbox: (x0, y0, x1, y1) in points, size}].
    bbox is None when the character could not be mapped to a glyph."""
    tables = list(_glyph_tables(doc, page).values())   # a file may hold several Mus2 subsets
    if not tables:
        return []
    syms = []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if MUSIC_FONT not in s["font"]:
                    continue
                for ch in s["chars"]:
                    code = _code(ch["c"])
                    if code is None or ch["c"].isspace():
                        continue
                    hit = next(((cm[k], gl, u) for cm, gl, u in tables
                                for k in (code, 0xF000 + code) if k in cm), None)
                    if hit is None:
                        syms.append(dict(code=code, bbox=None, size=s["size"])); continue
                    gname, glyf, upm = hit
                    g = glyf[gname]
                    if g.numberOfContours == 0:
                        continue            # empty glyph
                    ox, oy = ch["origin"]; z = s["size"] / upm
                    syms.append(dict(code=code, size=s["size"],
                                     bbox=(ox + g.xMin * z, oy - g.yMax * z, ox + g.xMax * z, oy - g.yMin * z)))
    return syms


def page_drawings(page) -> list[dict]:
    """Vector drawings: horizontal/vertical lines, curves (slurs), filled shapes (beams etc.)."""
    out = []
    for d in page.get_drawings():
        r = d["rect"]; kinds = {it[0] for it in d["items"]}
        if "c" in kinds: k = "curve"
        elif d["type"] in ("f", "fs"): k = "filled"
        elif r.height < 1.0 and r.width > 2: k = "hline"
        elif r.width < 1.0 and r.height > 2: k = "vline"
        else: k = "other"
        out.append(dict(kind=k, bbox=(r.x0, r.y0, r.x1, r.y1)))
    return out


def font_kind(doc) -> str:
    """How the file embeds the music font: 'truetype' (extracted directly by this
    module), 'cff_type0' (needs extraction by glyph index) or 'no_font' (symbols
    are vector paths, raster images or an unnamed font)."""
    kinds = set()
    for page in doc:
        for f in page.get_fonts(full=True):
            if MUSIC_FONT in f[3]:
                kinds.add("truetype" if (f[1] == "ttf" and f[2] == "TrueType") else "cff_type0")
    if "truetype" in kinds: return "truetype"
    return "cff_type0" if kinds else "no_font"
