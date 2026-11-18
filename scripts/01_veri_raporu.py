"""SymbTr v3 veri raporu: dağılımlar, sözlük boyutları, eser bazlı katmanlı bölme.
Çalıştırma: python scripts/01_veri_raporu.py"""
import os, sys, collections, statistics, random, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.symbtr import iter_pieces, REST
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
MIN_PIECES = 20          # tam değerlendirme için makam başına en az eser
SEED = 13

pieces = []
mk, us, fm = collections.Counter(), collections.Counter(), collections.Counter()
pitch, dur, kod = collections.Counter(), collections.Counter(), collections.Counter()
notes_mk = collections.Counter()
finalis = collections.defaultdict(collections.Counter)
for p in iter_pieces():
    ns = p.notes
    sounded = [n for n in ns if n[0] != REST]
    for r in p.rows: kod[r["Kod"]] += 1
    for n in sounded: pitch[n[0]] += 1
    for n in ns: dur[(n[1], n[2])] += 1
    m = p.meta
    mk[m["makam"]] += 1; us[m["usul"]] += 1; fm[m["form"]] += 1
    notes_mk[m["makam"]] += len(sounded)
    if sounded: finalis[m["makam"]][sounded[-1][0]] += 1
    pieces.append(dict(fname=p.fname, **m, n_note=len(sounded), n_rest=len(ns) - len(sounded),
                       n_row=len(p.rows)))

N = len(pieces); tot = sum(x["n_note"] for x in pieces)
lens = sorted(x["n_note"] for x in pieces)

# --- eser bazlı bölme: aynı eser adı + besteci aynı kümede kalır (sızıntı önlemi)
big = {m for m, c in mk.items() if c >= MIN_PIECES}
groups = collections.defaultdict(list)
for x in pieces:
    key = (x["eser"], x["besteci"]) if x["eser"] else (x["fname"],)
    groups[key].append(x)
rng = random.Random(SEED)
by_mk = collections.defaultdict(list)
for key, g in groups.items():
    by_mk[g[0]["makam"]].append(g)
for m, gl in by_mk.items():
    rng.shuffle(gl)
    n = len(gl)
    for i, g in enumerate(gl):
        if m not in big: s = "az_ornekli"
        elif i < round(0.8 * n): s = "egitim"
        elif i < round(0.9 * n): s = "dogrulama"
        else: s = "test"
        for x in g: x["bolme"] = s
with open(os.path.join(ROOT, "data", "bolmeler.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pieces[0].keys())); w.writeheader(); w.writerows(pieces)
split = collections.Counter(x["bolme"] for x in pieces)
split_notes = collections.Counter()
for x in pieces: split_notes[x["bolme"]] += x["n_note"]
multi = sum(1 for g in groups.values() if len(g) > 1)

# --- şekiller
fig, ax = plt.subplots(figsize=(8, 3.2))
vals = sorted(mk.values(), reverse=True)
ax.bar(range(len(vals)), vals, color="#3b6ea5"); ax.set_yscale("log")
ax.axhline(MIN_PIECES, color="#b33", lw=1, ls="--"); ax.text(len(vals) * 0.6, MIN_PIECES * 1.15, f"eşik = {MIN_PIECES} eser", color="#b33", fontsize=8)
ax.set_xlabel("makam (eser sayısına göre sıralı)"); ax.set_ylabel("eser sayısı (log)")
ax.set_title("SymbTr v3: makam başına eser sayısı"); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "reports", "figures", "makam_dagilimi.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist([l for l in lens if l <= 1500], bins=60, color="#3b6ea5")
ax.set_xlabel("eser başına sesli nota"); ax.set_ylabel("eser sayısı")
ax.set_title("SymbTr v3: eser uzunlukları (1500 üstü kırpıldı)"); fig.tight_layout()
fig.savefig(os.path.join(ROOT, "reports", "figures", "eser_uzunluklari.png"), dpi=150); plt.close(fig)

def thr(c, t): return sum(1 for v in c.values() if v < t)
def cover(c, t):
    k = [m for m, v in c.items() if v >= t]; return len(k), sum(c[m] for m in k)

L = []
A = L.append
A("# SymbTr v3 veri raporu\n")
A("`scripts/01_veri_raporu.py` tarafından üretilmiştir. Sayılar `symbtr_v3/txt_v3` içindeki dosyalardan hesaplanır; makam/form/usûl etiketleri dosya adından alınır.\n")
A("## Genel\n")
A(f"| Ölçü | Değer |\n|---|---|\n| Eser (dosya) | {N} |\n| Makam etiketi | {len(mk)} |\n| Usûl etiketi | {len(us)} |\n| Form etiketi | {len(fm)} |\n"
  f"| Sesli nota | {tot:,} |\n| Toplam satır (tüm Kod değerleri) | {sum(kod.values()):,} |\n"
  f"| Eser başına sesli nota (ortanca / ortalama) | {statistics.median(lens):.0f} / {statistics.mean(lens):.0f} |\n"
  f"| En kısa / en uzun eser | {lens[0]} / {lens[-1]} |\n"
  f"| Farklı perde (Koma53) | {len(pitch)} |\n| Farklı süre (Pay/Payda) | {len(dur)} |\n")
A("\n## Makam dağılımı (uzun kuyruk)\n")
A("![makam dağılımı](figures/makam_dagilimi.png)\n")
A("| Eşik | Eşiğin altındaki makam sayısı |\n|---|---|")
for t in (2, 5, 10, 20, 50): A(f"| < {t} eser | {thr(mk, t)} |")
A("\n| Eşik | Makam | Eser | Eser payı |\n|---|---|---|---|")
for t in (10, 20, 50, 100):
    k, c = cover(mk, t); A(f"| ≥ {t} eser | {k} | {c} | %{100 * c / N:.0f} |")
A(f"\n**En az {MIN_PIECES} eseri olan makamlar** (eser / sesli nota):\n")
A("| Makam | Eser | Nota | En sık bitiş perdesi (Koma53) | Bu perdede biten eser payı |\n|---|---|---|---|---|")
for m, c in mk.most_common():
    if c < MIN_PIECES: break
    fp, fc = finalis[m].most_common(1)[0]
    A(f"| {m} | {c} | {notes_mk[m]:,} | {fp} | %{100 * fc / sum(finalis[m].values()):.0f} |")
A("\n## Usûl ve form\n")
A(f"- 5'ten az eseri olan usûl: {thr(us, 5)} / {len(us)}; en az 20 eseri olan usûl: {cover(us, 20)[0]} (eserlerin %{100 * cover(us, 20)[1] / N:.0f}'i).")
A("- En sık usûller: " + ", ".join(f"{k} ({v})" for k, v in us.most_common(10)) + ".")
A("- En sık formlar: " + ", ".join(f"{k} ({v})" for k, v in fm.most_common(10)) + ".")
A("\n## Eser uzunlukları\n")
A("![eser uzunlukları](figures/eser_uzunluklari.png)\n")
A("\n## Sözlük\n")
A(f"- En sık 20 süre değeri tüm nota/susların %{100 * sum(v for _, v in dur.most_common(20)) / sum(dur.values()):.1f}'ini kapsıyor ({len(dur)} farklı değerden).")
A(f"- Perde aralığı (Koma53): {min(pitch)}–{max(pitch)}; en sık 60 perde sesli notaların %{100 * sum(v for _, v in pitch.most_common(60)) / tot:.1f}'ini kapsıyor.")
A("- Satır türleri (Kod): " + ", ".join(f"{k}: {v:,}" for k, v in kod.most_common(8)) + ".")
A("\n## Eser bazlı katmanlı bölme\n")
A(f"`data/bolmeler.csv` (tohum {SEED}). Kural: en az {MIN_PIECES} eserli makamlarda makam içinde 80/10/10; diğer makamlar `az_ornekli` havuzunda (az örnekli genelleme deneyleri için). Aynı eser adı ve besteciye sahip dosyalar ({multi} grup) aynı kümede tutulur.\n")
A("| Küme | Eser | Sesli nota |\n|---|---|---|")
for s in ("egitim", "dogrulama", "test", "az_ornekli"): A(f"| {s} | {split[s]} | {split_notes[s]:,} |")
A("\n## Notlar\n")
A("- Dosyalar karışık kodlamalı (UTF-8 ve Windows-1254); ayrıştırıcı ikisini de dener.")
A("- Dosya adındaki etiketler Zenodo kaydında bildirilen sayılarla (164 makam, 135 usûl, 61 form) birebir örtüşmüyor; fark etiket varyantlarından (örn. `hicaz_uzzal`, `yuruksemai_4`) ve usûl değişimli eserlerden kaynaklanıyor olabilir, ayrıca incelenmeli.")
A("- txt dosyaları tekrarları açılmış icra sırasını içerir; basılı notadaki tekrar işaretleriyle birebir hizalama için mu2/MusicXML sürümleri gerekir.")
open(os.path.join(ROOT, "reports", "veri_raporu.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
