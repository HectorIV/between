"""Assets de fondo para la landing: composicion panoramic con la luna."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

os.makedirs("assets", exist_ok=True)
CW, CH = 495, 760

def lay(n):
    return Image.open(f"capas/{n}.png").convert("RGBA")

moon_s = lay("03_luna")
halo_s = lay("01_halo_resplandor")
rays_s = lay("02_rayos_luz")
spark_s = lay("08_chimpas") if False else lay("08_chispas")
frame_s = lay("04_marco_ornamental")


def place(canvas, img, cx, cy, target_h, opa=1.0):
    """Coloca img centrada en (cx,cy) con la altura objetivo, respetando alfa."""
    s = target_h / img.height
    w = max(2, int(img.width * s))
    h = max(2, int(img.height * s))
    r = img.resize((w, h), Image.LANCZOS)
    if opa != 1.0:
        a = r.getchannel("A").point(lambda v: int(v * opa))
        r.putalpha(a)
    out = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    out.paste(r, (int(cx - w / 2), int(cy - h / 2)), r)
    return Image.alpha_composite(canvas, out)


def vignette(arr, strength=0.78, power=1.8):
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    v = np.sqrt(((xx - W/2)/(W/2))**2 + ((yy - H/2)/(H/2))**2)
    m = np.clip(1 - strength * np.clip(v, 0, 1) ** power, 0.05, 1.0)
    arr[..., :3] *= m[..., None]
    return arr


def grain(arr, amt=3.2, seed=11):
    rs = np.random.default_rng(seed)
    n = rs.normal(0, amt, arr.shape[:2]).astype(np.float32)
    return arr + n[..., None]


def to_rgb(img_or_arr):
    """RGBA(float|uint8) -> RGB uint8. Importante: sin pasar mode='RGB'
    a un array de 4 canales, Pillow desalinea los canales."""
    a = img_or_arr if isinstance(img_or_arr, np.ndarray) else np.asarray(img_or_arr)
    a = a[..., :3] if a.shape[-1] == 4 else a
    return np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))


# ============================================================ HERO 1920x1080
BW, BH = 1920, 1080
yy, xx = np.mgrid[0:BH, 0:BW]
bg = np.zeros((BH, BW, 4), np.float32)
bg[..., 3] = 255
hero = Image.fromarray(bg.astype(np.uint8), "RGBA")

# resplandor amplio de fondo
rad = np.sqrt(((xx - BW/2)/(BW*0.34))**2 + ((yy - BH*0.48)/(BH*0.78))**2)
g = np.clip(1 - rad, 0, 1) ** 1.9
tint = np.dstack([10 + 52*g, 8 + 36*g, 11 + 19*g, np.full((BH, BW), 255)])
hero = Image.fromarray(tint.astype(np.uint8), "RGBA")

MCX, MCY, MH = BW/2, BH*0.46, 1000
hero = place(hero, halo_s, MCX, MCY, MH, 1.0)
hero = place(hero, rays_s, MCX, MCY, MH, 1.0)
hero = place(hero, moon_s, MCX, MCY, MH, 1.0)
hero = place(hero, spark_s, MCX, MCY, int(MH*1.1), 0.85)

# orla dorada en los cuatro bordes (recorte de las columnas del marco)
for x0, fl in [(0, False), (BW-150, True)]:
    col = frame_s.crop((fl and 355 or 0, 0, (fl and 495 or 140), 760))
    col = col.resize((150, BH+200), Image.LANCZOS)
    a = col.getchannel("A").point(lambda v: int(v*0.9))
    col.putalpha(a)
    tmp = Image.new("RGBA", hero.size, (0,0,0,0))
    tmp.paste(col, (x0, -100), col)
    hero = Image.alpha_composite(hero, tmp)

# cenefa dorada superior e inferior
rule = Image.new("RGBA", (BW, BH), (0,0,0,0))
d = ImageDraw.Draw(rule)
d.line([(0, 0), (BW, 0)], fill=(150, 106, 34, 46), width=2)
d.line([(0, BH-1), (BW, BH-1)], fill=(150, 106, 34, 46), width=2)
hero = Image.alpha_composite(hero, rule)

arr = np.asarray(hero).astype(np.float32)
arr[..., :3] = np.clip(arr[..., :3] * 1.22, 0, 255)   # exposure
arr = vignette(arr, 0.62, 2.2)
# banda inferior algo mas oscura para el CTA del hero
band = np.clip((yy - BH*0.50) / (BH*0.34), 0, 1) ** 1.4
arr[..., :3] *= (1 - 0.30*band)[..., None]
arr = grain(arr, 2.6)
Image.fromarray(to_rgb(arr)).save(
    "assets/hero_bg.jpg", quality=90, optimize=True)
print("hero_bg.jpg ok")


# ================================================= FONDO SECCION OSCURA
DW, DH = 1600, 900
yy2, xx2 = np.mgrid[0:DH, 0:DW]
base = np.zeros((DH, DW, 3), np.float32)
base[..., 0] = 5
base[..., 1] = 6
base[..., 2] = 11
dark = Image.fromarray(base.astype(np.uint8), "RGB").convert("RGBA")
# luna fria a la derecha
cold = moon_s.copy()
r_, g_, b_, a_ = cold.split()
hsv = Image.merge("RGB", (r_, g_, b_)).convert("HSV")
h, s, v = hsv.split()
ch = Image.new("RGB", cold.size, (104, 142, 196)).convert("HSV").split()[0]
h = Image.blend(h, ch, 0.80)
cold = Image.merge("HSV", (h, s, v)).convert("RGB")
cold.putalpha(a_)
dark = place(dark, halo_s, DW*0.74, DH*0.46, 760, 0.42)
dark = place(dark, cold, DW*0.74, DH*0.46, 700, 0.85)
dark = place(dark, spark_s, DW*0.74, DH*0.46, 820, 0.45)
arr = np.asarray(dark).astype(np.float32)
arr[..., :3] *= 0.62
arr = vignette(arr, 0.72, 1.7)
arr = grain(arr, 2.2, seed=23)
Image.fromarray(to_rgb(arr)).save(
    "assets/bg_dark.jpg", quality=88, optimize=True)
print("bg_dark.jpg ok")


# ================================================= FONDO SECCION LUZ (clara)
light = np.zeros((DH, DW, 3), np.float32)
rr = np.sqrt(((xx2 - DW*0.30)/(DW*0.42))**2 + ((yy2 - DH*0.45)/(DH*0.75))**2)
gl = np.clip(1 - rr, 0, 1) ** 2.0
light[..., 0] = 14 + 120*gl
light[..., 1] = 11 + 84*gl
light[..., 2] = 10 + 44*gl
lg = Image.fromarray(np.clip(light, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
lg = place(lg, halo_s, DW*0.30, DH*0.45, 780, 0.55)
arr = np.asarray(lg).astype(np.float32)
arr = vignette(arr, 0.62, 1.6)
arr = grain(arr, 2.0, seed=31)
Image.fromarray(to_rgb(arr)).save(
    "assets/bg_light.jpg", quality=88, optimize=True)
print("bg_light.jpg ok")
