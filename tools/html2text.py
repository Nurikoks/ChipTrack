"""Strip a downloaded legal-act HTML page to plain text (one block per line).

Usage: python -I html2text.py <in.html> <out.txt>
"""
import html
import re
import sys

raw = open(sys.argv[1], "rb").read()
for enc in ("utf-8", "cp1251"):
    try:
        s = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", s)
s = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d|table)>", "\n", s)
s = re.sub(r"(?s)<[^>]+>", " ", s)
s = html.unescape(s)
lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in s.splitlines()]
text = "\n".join(ln for ln in lines if ln)
open(sys.argv[2], "w", encoding="utf-8").write(text)
print(sys.argv[2], len(text), "chars")
