"""Optimiza los recursos de la landing para poder enviarla por correo."""
import os
import numpy as np
from PIL import Image

OUT = "assets/opt"
os.makedirs(OUT, exist_ok=True)


def kb(p):
    return os.path.getsize(p) / 1024


# ---- hero_bg: JPEG -> WebP (el navegador lo soporta desde 2020)
src = Image.open("assets/hero_bg.jpg").convert("RGB")
for q in (80, 86):
    src.save(f"{OUT}/hero_bg.webp", "WEBP", quality=q, method=6)
print("hero_bg")
print(f"   original jpg  {kb('assets/hero_bg.jpg'):8.1f} KB")
for q in (80, 86):
    print(f"   webp q{q:<3}     {kb(f'{OUT}/hero_bg.webp') if q==86 else '-':>8} KB")

# ---- poster del video: PNG 473 KB -> JPEG pequeno
p = Image.open("anim/poster.png").convert("RGB")
print(f"poster original  {kb('anim/poster.png'):8.1f} KB  {p.size}")
p.resize((496, 760), Image.LANCZOS).save(f"{OUT}/poster.jpg", "JPEG",
                                        quality=82, optimize=True, progressive=True)
print(f"   poster.jpg    {kb(f'{OUT}/poster.jpg'):8.1f} KB")

# thumbnail aun mas pequena para el <video poster> (se ve un instante)
p.resize((360, 552), Image.LANCZOS).save(f"{OUT}/poster_sm.jpg", "JPEG",
                                         quality=72, optimize=True, progressive=True)
print(f"   poster_sm.jpg {kb(f'{OUT}/poster_sm.jpg'):8.1f} KB")

# ---- mariposas: PNG con alfa -> WebP con alfa
for i in (1, 2, 3):
    f = f"assets/mariposa{i}.png"
    im = Image.open(f).convert("RGBA")
    im.save(f"{OUT}/mariposa{i}.webp", "WEBP", quality=88, method=6)
    print(f"   mariposa{i}    png {kb(f):7.1f} KB -> webp {kb(f'{OUT}/mariposa{i}.webp'):7.1f} KB")

# ---- logo: ya es pequeno
lg = Image.open("assets/logo.png").convert("RGBA")
lg.save(f"{OUT}/logo.webp", "WEBP", quality=90, method=6)
print(f"   logo          png {kb('assets/logo.png'):7.1f} KB -> webp {kb(f'{OUT}/logo.webp'):7.1f} KB")

print("\noptimizados en", OUT)
