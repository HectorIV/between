import re, os, io

s = io.open("landing.html", encoding="utf-8").read()

# cualquier atributo/valor que parezca una ruta local terminada en extensión
pat = re.compile(r'["\'\s(]([\w./-]+\.(?:png|jpe?g|webp|gif|mp4|svg|woff2?))["\'\s)]', re.I)
refs = set(m.group(1) for m in pat.finditer(s))

ext = sorted(r for r in refs if r.startswith(("http", "//")) or "fonts.g" in r)
loc = sorted(r for r in refs if r not in ext)

print("=== HTML ===")
h = os.path.getsize("landing.html") / 1024
print(f"{h:9.1f} KB  landing.html")
print()
print("=== recursos locales ===")
tot = 0.0
for f in loc:
    if os.path.exists(f):
        k = os.path.getsize(f) / 1024
        tot += k
        print(f"{k:9.1f} KB  {f}")
    else:
        print(f"   FALTA       {f}")
print(f"{tot:9.1f} KB  subtotal")
print(f"{h + tot:9.1f} KB  TOTAL crudo  ({((h+tot)/1024):.2f} MB)")
print()
print("=== externos ===")
for e in ext:
    print("  ", e[:95])
