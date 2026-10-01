import os, glob
import numpy as np
from PIL import Image

fr = sorted(glob.glob("anim/frames/*.png"))
imgs = [Image.open(f).convert("RGB") for f in fr]

# GIF: reducir resolucion y usar paleta global para evitar parpadeo de color
S = 0.72
tw, th = int(495*S)//2*2, int(760*S)//2*2   # pares para compatibilidad
small = [im.resize((tw, th), Image.LANCZOS) for im in imgs]

# paleta global unica -> sin "color breathing" entre frames
pal = small[0].quantize(colors=192, method=Image.MEDIANCUT)
q = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in small]
q[0].save("anim/BETWEEN_animada.gif", save_all=True, append_images=q[1:],
          duration=int(1000/24), loop=0, optimize=True, disposal=1)
print("gif", os.path.getsize("anim/BETWEEN_animada.gif")//1024, "KB", q[0].size)

# version webp 2x para calidad
big = [im.resize((990, 1520), Image.LANCZOS) for im in imgs]
big[0].save("anim/BETWEEN_animada_HD.webp", save_all=True, append_images=big[1:],
            duration=int(1000/24), loop=0, quality=90, method=4)
print("webp HD", os.path.getsize("anim/BETWEEN_animada_HD.webp")//1024, "KB")
