import os, glob
import numpy as np
from PIL import Image

fr = sorted(glob.glob("anim/frames/*.png"))
print("frames:", len(fr))
imgs = [Image.open(f).convert("RGB") for f in fr]
W, H = imgs[0].size

# poster: frame mas "limpio" (indice medio)
imgs[len(imgs)//2].save("anim/poster.png")

# --- GIF optimizado (paleta) ---
gifs = []
for im in imgs:
    g = im.quantize(colors=200, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    gifs.append(g)
gifs[0].save("anim/BETWEEN_animada.gif", save_all=True, append_images=gifs[1:],
             duration=int(1000/24), loop=0, optimize=True, disposal=2)
print("gif ok", os.path.getsize("anim/BETWEEN_animada.gif")//1024, "KB")

# --- WebP animado (mejor calidad/tamano) ---
imgs[0].save("anim/BETWEEN_animada.webp", save_all=True, append_images=imgs[1:],
             duration=int(1000/24), loop=0, quality=88, method=4)
print("webp ok", os.path.getsize("anim/BETWEEN_animada.webp")//1024, "KB")
