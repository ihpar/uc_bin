"""Build the single-file test page: python feasibility/vexflow/build.py  ->  dist/index.html
Needs `npm install` in this folder first (VexFlow is inlined from node_modules)."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
vf = open(os.path.join(H, "node_modules", "vexflow", "build", "cjs", "vexflow.js"), encoding="utf-8").read()
pieces = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(H, "pieces", "*.json")))]
pieces.sort(key=lambda p: p["notes"])
html = open(os.path.join(H, "template.html"), encoding="utf-8").read()
html = html.replace("/*VEXFLOW*/", vf.replace("</script", "<\\/script"))
html = html.replace("/*PIECES*/", json.dumps(pieces, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
html = html.replace("/*APP*/", open(os.path.join(H, "app.js"), encoding="utf-8").read())
os.makedirs(os.path.join(H, "dist"), exist_ok=True)
open(os.path.join(H, "dist", "index.html"), "w", encoding="utf-8").write(html)
print("dist/index.html", round(len(html) / 1e6, 2), "MB,", len(pieces), "pieces")
