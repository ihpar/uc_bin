"""SymbTr txt -> JSON for the VexFlow feasibility test.

    python feasibility/vexflow/make_piece.py <symbtr file name without .txt> [...]

Output: pieces/<name>.json with measures of notes; every note has a staff
position (key), an AEU accidental (e.g. b1, #4) or None, and a duration that a
single note symbol can show. Longer or irregular durations are split into tied
notes, which is what an editor has to do anyway."""
import collections, json, os, re, sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from symbtr import load_piece, split_at_bar_lines, read_text, TXT_DIR, MU2_DIR   # noqa: E402

NATURAL = dict(C=0, D=9, E=18, F=22, G=31, A=40, B=49)      # Holdrian commas above C


def key_signature_53(name):
    """Key signature in numbered commas, from the mu2 file (row code 50)."""
    path = os.path.join(MU2_DIR, name + ".mu2")
    if not os.path.exists(path):
        return []
    for line in read_text(path).splitlines():
        r = [c.strip() for c in line.split("\t")]
        if r[0] == "50" and len(r) > 8:
            out = []
            for k in filter(None, r[8].split("/")):
                m = re.match(r"^([A-G])(\d)([b#]\d)$", k)
                if m: out.append([m.group(3), f"{m.group(1).lower()}/{m.group(2)}"])
            return out
    return []


# single symbols: (fraction of a whole note, VexFlow duration, dots)
SIMPLE = [(Fraction(1, d) * m, str(d), dots) for d in (1, 2, 4, 8, 16, 32, 64)
          for m, dots in ((Fraction(7, 4), 2), (Fraction(3, 2), 1), (Fraction(1), 0))]
SIMPLE.sort(reverse=True)


def pieces_of(d: Fraction):
    """Split a duration into single-symbol durations, longest first."""
    out = []
    while d > 0:
        for f, name, dots in SIMPLE:
            if f <= d:
                out.append((name, dots)); d -= f
                break
        else:
            return None
    return out


def convert(name):
    p = load_piece(name + ".txt", TXT_DIR)
    events = split_at_bar_lines(p.events)
    bars = collections.defaultdict(list)
    skipped = 0
    for e in events:
        d = Fraction(e["num"], e["den"])
        tuplet = d.denominator % 3 == 0
        parts = pieces_of(d * Fraction(3, 2) if tuplet else d)
        if parts is None:
            skipped += 1
            continue
        m = re.match(r"^([A-G])(\d)(?:([b#])(\d))?$", e["ae"] or "")
        rest = e["koma"] == -1 or not m
        for i, (dur, dots) in enumerate(parts):
            n = dict(d=dur)
            if dots: n["dots"] = dots
            if rest: n["rest"] = 1
            else:
                n["k"] = f"{m.group(1).lower()}/{m.group(2)}"
                if m.group(3): n["acc"] = m.group(3) + m.group(4)
                # the same note in the numbered-comma spelling (SymbTr column Koma53)
                off = e["koma"] - (318 + NATURAL[m.group(1)] + 53 * (int(m.group(2)) - 5))
                a53 = ("#" if off > 0 else "b") + str(abs(off)) if off else None
                if abs(off) <= 9 and a53 != n.get("acc"): n["a53"] = a53 or "n"
                if i > 0 or e.get("tie"): n["tie"] = 1
            if tuplet: n["tup"] = 1
            bars[int(e["onset"] + 1e-9)].append(n)
    lens = collections.Counter()
    for k in bars:
        lens[sum((Fraction(e["num"], e["den"]) for e in events if int(e["onset"] + 1e-9) == k), Fraction(0))] += 1
    meter = lens.most_common(1)[0][0]
    den = 8 if meter * 8 <= 24 else 4
    # A measure of a long usul (hafif: 32 half notes) is wider than a line. It is cut into
    # segments of about one whole note at note boundaries; "c" marks a continuation, which
    # is drawn with a dashed bar line as in the printed scores.
    measures = []
    for k in sorted(bars):
        seg, acc, first = [], Fraction(0), True
        for n in bars[k]:
            seg.append(n)
            acc += Fraction(1, int(n["d"])) * (Fraction(2 ** (n.get("dots", 0) + 1) - 1, 2 ** n.get("dots", 0))) \
                * (Fraction(2, 3) if n.get("tup") else 1)
            if meter > 3 and acc >= 1 and acc == int(acc):
                measures.append(dict(n=seg) if first else dict(n=seg, c=1)); seg, acc, first = [], Fraction(0), False
        if seg:
            measures.append(dict(n=seg) if first else dict(n=seg, c=1))
    out = dict(name=name, **p.meta, time=[int(meter * den), den], keysig53=key_signature_53(name),
               measures=measures, skipped=skipped)
    # the same key signature in AEU spelling: what the notes it applies to are called there
    out["keysig"] = []
    for a53, pitch in out["keysig53"]:
        seen = collections.Counter(n.get("acc") for b in measures for n in b["n"]
                                   if not n.get("rest") and n["k"][0] == pitch[0] and n.get("a53", n.get("acc")) == a53)
        out["keysig"].append([seen.most_common(1)[0][0] if seen else a53, pitch])
    out["notes"] = sum(len(b["n"]) for b in measures)
    with open(os.path.join(HERE, "pieces", name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(name, "measures", len(out["measures"]), "symbols", out["notes"], "time", out["time"], "skipped", skipped)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        convert(n)
