import os, glob
import numpy as np
from PIL import Image, ImageDraw

files = sorted(glob.glob("capas/*.png"))
cols = 5
SC = 0.62
th = int(760*SC); tw = int(495*SC)
rows = (len(files)+cols-1)//cols
pad = 22
sheet = Image.new("RGB", (cols*(tw+pad)+pad, rows*(th+pad+18)+pad), (24,24,28))
d = ImageDraw.Draw(sheet)
for i,f in enumerate(files):
    lay = Image.open(f).convert("RGBA")
    lay = lay.resize((tw,th), Image.LANCZOS)
    # checker
    ch = Image.new("RGBA",(tw,th),(0,0,0,0))
    dd = ImageDraw.Draw(ch)
    s=10
    for y in range(0,th,s):
        for x in range(0,tw,s):
            c = (60,60,66,255) if ((x//s+y//s)%2==0) else (40,40,46,255)
            dd.rectangle([x,y,x+s-1,y+s-1], fill=c)
    ch.alpha_composite(lay)
    cx = pad + (i%cols)*(tw+pad); cy = pad + (i//cols)*(th+pad+18)
    sheet.paste(ch.convert("RGB"), (cx,cy))
    d.text((cx+2, cy+th+3), os.path.basename(f)[:-4], fill=(230,230,235))
sheet.save("capas/_CONTACTO.png")
print("contacto", sheet.size, len(files), "capas")
