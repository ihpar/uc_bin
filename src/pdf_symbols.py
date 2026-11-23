"""Automatic symbol bounding boxes from SymbTr PDFs (vector, embedded Mus2 font).

Every music symbol is a character of the Mus2 font. The character position on
the page comes from the PDF; the tight bounding box comes from the glyph bounds
of the embedded font. Staff lines, barlines, beams, slurs etc. are vector
drawings rather than font characters and are returned by `page_drawings`."""
from __future__ import annotations
import io, logging
import pymupdf
from fontTools.cffLib import CFFFontSet
from fontTools.ttLib import TTFont

logging.getLogger("fontTools").setLevel(logging.ERROR)
MUSIC_FONT = "Mus2"


def _glyph_tables(doc, page) -> dict:
    """{font name: lookup} for the music fonts on the page, where lookup(code)
    returns (xMin, yMin, xMax, yMax, unitsPerEm) of the glyph, () for an empty
    glyph, or None when the font has no glyph for that code.

    Two embeddings occur in SymbTr: TrueType subsets (glyph bounds from the glyf
    table) and bare CFF Type 1 subsets written by Acrobat Distiller (glyph bounds
    from the charstrings, character codes from the font's built-in encoding)."""
    out = {}
    for f in page.get_fonts(full=True):
        if MUSIC_FONT not in f[3]:
            continue
        try:
            buf = doc.extract_font(f[0])[3]
        except Exception:
            buf = None
        if not buf:
            continue
        if f[1] == "ttf" and f[2] == "TrueType":
            try:
                t = TTFont(io.BytesIO(buf))
            except Exception:
                continue
            cmap = {}
            for tb in t["cmap"].tables:
                cmap.update(tb.cmap)
            out[f[3]] = _ttf_lookup(cmap, t["glyf"], t["head"].unitsPerEm)
        elif f[1] in ("cff", "cid"):
            try:
                cff = CFFFontSet(); cff.decompile(io.BytesIO(buf), None)
                top = cff[0]
                upm = round(1 / top.FontMatrix[0]) if getattr(top, "FontMatrix", None) else 1000
                if hasattr(top, "ROS"):         # CID-keyed (Type 0, Identity-H): glyph name is the code
                    enc = _CidNames()
                else:
                    enc = top.Encoding
                    if isinstance(enc, str):    # predefined encoding: not seen in SymbTr
                        continue
                out[f[3]] = _cff_lookup(enc, top.CharStrings, upm)
            except Exception:
                continue
    return out


class _CidNames:
    """Encoding of a CID-keyed font: code n is the glyph named cid<n>."""
    def __len__(self): return 65536
    def __getitem__(self, code): return f"cid{code:05d}"


def _ttf_lookup(cmap, glyf, upm):
    def look(code):
        name = next((cmap[k] for k in (code, 0xF000 + code) if k in cmap), None)
        if name is None:
            return None
        g = glyf[name]
        return () if g.numberOfContours == 0 else (g.xMin, g.yMin, g.xMax, g.yMax, upm)
    return look


def _cff_lookup(enc, charstrings, upm):
    cache = {}
    def look(code):
        if code not in cache:
            name = enc[code] if code < len(enc) else ".notdef"
            if name == ".notdef" or name not in charstrings:
                cache[code] = None
            else:
                b = charstrings[name].calcBounds(charstrings)
                cache[code] = () if b is None else (b[0], b[1], b[2], b[3], upm)
        return cache[code]
    return look


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
                    hit = next((g for g in (look(code) for look in tables) if g is not None), None)
                    if hit is None:
                        syms.append(dict(code=code, bbox=None, size=s["size"])); continue
                    if not hit:
                        continue            # empty glyph
                    xmin, ymin, xmax, ymax, upm = hit
                    ox, oy = ch["origin"]; z = s["size"] / upm
                    syms.append(dict(code=code, size=s["size"],
                                     bbox=(ox + xmin * z, oy - ymax * z, ox + xmax * z, oy - ymin * z)))
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
    """How the file embeds the music font: 'truetype' or 'cff' (both read by this
    module), 'other' (a named Mus2 font in another embedding) or 'no_font'
    (symbols are vector paths, raster images or an unnamed font)."""
    kinds = set()
    for page in doc:
        for f in page.get_fonts(full=True):
            if MUSIC_FONT in f[3]:
                kinds.add("truetype" if (f[1] == "ttf" and f[2] == "TrueType") else
                          "cff" if f[1] in ("cff", "cid") else "other")
    for k in ("truetype", "cff", "other"):
        if k in kinds: return k
    return "no_font"
