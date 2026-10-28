"""Build one self-contained HTML file from the project, for publishing as a claude.ai artifact.

The project (index.html + css/ + js/) is the source of truth. The artifact host wraps the page in its
own <html>/<head>/<body>, so the output starts with <title> and <style> and inlines every script in the
order index.html loads them.

    python tools/build_artifact.py            -> dist/sdf-walkers.html
"""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent
index = (root / "index.html").read_text(encoding="utf-8")

head = re.search(r"<head>(.*?)</head>", index, re.S).group(1)
body = re.search(r"<body>(.*?)</body>", index, re.S).group(1)

keep_head = [line for line in head.strip().splitlines()
             if line.startswith(("<title", '<meta name="claude:', '<link rel="preconnect"'))
             or ("fonts.googleapis.com/css2" in line)]
css = (root / "css" / "style.css").read_text(encoding="utf-8")

scripts = re.findall(r'<script src="([^"]+)"></script>', body)
markup = re.sub(r'\s*<script src="[^"]+"></script>', "", body).strip()
js = "\n".join((root / s).read_text(encoding="utf-8") for s in scripts)

out = "\n".join(keep_head) + f"\n<style>\n{css}</style>\n\n{markup}\n\n<script>\n{js}</script>\n"
dist = root / "dist"
dist.mkdir(exist_ok=True)
(dist / "sdf-walkers.html").write_text(out, encoding="utf-8")
print(f"dist/sdf-walkers.html  {len(out):,} bytes, {len(scripts)} scripts inlined")
