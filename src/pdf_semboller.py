"""SymbTr PDF'lerinden (vektör, gömülü Mus2 fontu) otomatik sembol kutusu çıkarımı.

Her müzik sembolü Mus2 fontunun bir karakteridir. Karakterin sayfadaki konumu
PDF'ten, sıkı sınır kutusu ise gömülü fontun glif sınırlarından hesaplanır.
Dizek çizgileri, ölçü çizgileri, kirişler, bağlar gibi öğeler font karakteri
değil vektör çizimdir; `page_drawings` ile ayrıca alınır."""
from __future__ import annotations
import io, logging
import pymupdf
from fontTools.ttLib import TTFont

logging.getLogger("fontTools").setLevel(logging.ERROR)
MUSIC_FONT = "Mus2"


def _glyph_boxes(doc, page) -> dict:
    """Sayfadaki müzik fontları için {font_adı: (cmap, glyf, upm)}."""
    out = {}
    for f in page.get_fonts(full=True):
        if MUSIC_FONT not in f[3]:
            continue
        buf = doc.extract_font(f[0])[3]
        if not buf:
            continue
        if f[1] != "ttf" or f[2] != "TrueType":
            continue                # CFF/Type0 gömmeler: bkz. font_kinds()
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
    """MuPDF'in Unicode'a çevirdiği karakteri fonttaki bayt koduna geri çevirir."""
    try:
        return ch.encode("cp1252")[0]
    except (UnicodeEncodeError, IndexError):
        return ord(ch) if ord(ch) < 256 else None


def page_symbols(doc, page) -> list[dict]:
    """[{code, bbox:(x0,y0,x1,y1) punto cinsinden, size}] döndürür."""
    fonts = _glyph_boxes(doc, page)
    if not fonts:
        return []
    tables = list(fonts.values())      # dosyada birden çok Mus2 alt kümesi olabilir
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
                        continue            # boşluk
                    ox, oy = ch["origin"]; z = s["size"] / upm
                    syms.append(dict(code=code, size=s["size"],
                                     bbox=(ox + g.xMin * z, oy - g.yMax * z, ox + g.xMax * z, oy - g.yMin * z)))
    return syms


def page_drawings(page) -> list[dict]:
    """Vektör çizimler: yatay/dikey çizgi, eğri (bağ), dolu çokgen (kiriş vb.)."""
    out = []
    for d in page.get_drawings():
        r = d["rect"]; kinds = {it[0] for it in d["items"]}
        if "c" in kinds: k = "egri"
        elif d["type"] in ("f", "fs"): k = "dolu"
        elif r.height < 1.0 and r.width > 2: k = "yatay_cizgi"
        elif r.width < 1.0 and r.height > 2: k = "dikey_cizgi"
        else: k = "diger"
        out.append(dict(kind=k, bbox=(r.x0, r.y0, r.x1, r.y1)))
    return out


def font_kinds(doc) -> str:
    """Dosyanın müzik fontunu nasıl gömdüğünü sınıflar: 'truetype' (bu modülle
    doğrudan çıkarılır), 'cff_type0' (glif indeksiyle çıkarım gerekir) veya
    'font_yok' (semboller vektör yol/raster ya da adsız font)."""
    kinds = set()
    for page in doc:
        for f in page.get_fonts(full=True):
            if MUSIC_FONT in f[3]:
                kinds.add("truetype" if (f[1] == "ttf" and f[2] == "TrueType") else "cff_type0")
    if "truetype" in kinds: return "truetype"
    return "cff_type0" if kinds else "font_yok"
