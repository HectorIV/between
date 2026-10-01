"""
Extracción por capas de Between.png  ->  capas/*.png  (RGBA, mismo tamaño 495x760)

Orden de apilado (de atrás hacia delante):
  00 fondo negro
  01 halo / resplandor
  02 rayos de luz
  03 luna
  04 marco ornamental
  05 mariposas
  06 chica
  07 vestido turquesa (detalle)
  08 chispas
  09 titulo  10 autora  11 rombo  12 logo
"""
import os
import numpy as np
import cv2
from PIL import Image

SRC, OUT = "Between.png", "capas"
os.makedirs(OUT, exist_ok=True)

im = Image.open(SRC).convert("RGB")
RGB = np.asarray(im)
H, W = RGB.shape[:2]
A = RGB.astype(np.float32) / 255.0
r, g, b = A[..., 0], A[..., 1], A[..., 2]
lum = 0.299 * r + 0.587 * g + 0.114 * b
mx, mn = A.max(2), A.min(2)
sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
YY, XX = np.mgrid[0:H, 0:W]
CX, CY, R = 243.0, 232.0, 127.0
dl = np.sqrt((XX - CX) ** 2 + (YY - CY) ** 2)
ang = np.arctan2(YY - CY, XX - CX)
NA = 1440
ai = np.mod(np.floor((ang + np.pi) / (2 * np.pi) * NA).astype(int), NA)

gray = (np.clip(lum, 0, 1) * 255).astype(np.uint8)


def tophat(k):
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)


# ------------------------------------------------------------------ helpers
def blur(a, s):
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)


def op(m, k, it=1, o=cv2.MORPH_OPEN):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = m.astype(np.uint8)
    for _ in range(it):
        m = cv2.morphologyEx(m, o, ker)
    return m.astype(np.float32)


def fill_small_holes(m, maxa=400):
    m = m.astype(np.uint8).copy()
    ff = np.zeros((H + 2, W + 2), np.uint8)
    tmp = m.copy()
    cv2.floodFill(tmp, ff, (0, 0), 1)
    n, lbl, st, _ = cv2.connectedComponentsWithStats((tmp == 0).astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, 4] <= maxa:
            m[lbl == i] = 1
    return m.astype(np.float32)


def largest_cc(m, near=None):
    m = m.astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros((H, W), np.float32)
    i = int(lbl[near[1], near[0]]) if near else 0
    if i == 0:
        i = 1 + int(np.argmax(st[1:, 4]))
    return (lbl == i).astype(np.float32)


def keep_cc(m, minarea=30, drop_solid=True, frac=0.5, solid_min=700,
            minw=0, minh=0, maxarea=10 ** 9):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < minarea or area > maxarea:
            continue
        if w < minw or h < minh:
            continue
        if drop_solid and area > frac * w * h and area > solid_min:
            continue
        keep[i] = True
    return keep[lbl].astype(np.float32)


def save(name, rgb, alpha, boost=1.0):
    a8 = (np.clip(alpha * boost, 0, 1) * 255 + 0.5).astype(np.uint8)
    c8 = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([c8, a8]), "RGBA").save(os.path.join(OUT, name + ".png"))
    print(f"  {name:24s} px={int((alpha>0.05).sum()):6d}  maxA={alpha.max():.2f}")


# ================================================================= 04 MARCO
girlbox = np.zeros((H, W), bool)
girlbox[132:548, 136:372] = True
frame_zone = ~girlbox

mean = cv2.blur(gray.astype(np.float32), (5, 5))
std = np.sqrt(np.maximum(cv2.blur(gray.astype(np.float32) ** 2, (5, 5)) - mean ** 2, 0))
warm_sat = (sat > 0.50) & (r > b + 0.14) & (r > 0.16)
m_frame = (warm_sat & (std > 13) & frame_zone).astype(np.float32)
m_frame[dl < 131] = 0
m_frame = op(m_frame, 5, 1, cv2.MORPH_CLOSE)
m_frame = op(m_frame, 3, 1, cv2.MORPH_OPEN)
m_frame = keep_cc(m_frame, 30)
m_frame = fill_small_holes(m_frame, 300)
m_frame = blur(m_frame, 0.5)

# ================================================================= 03 LUNA
moon_a = np.clip((R - dl) / 2.4 + 0.5, 0, 1)
t = np.clip(1.0 - dl / R, 0, 1) ** 0.42
moon_rgb = (np.array([1.00, 0.905, 0.735], np.float32) * (1 - t[..., None]) +
            np.array([1.00, 0.985, 0.945], np.float32) * t[..., None]).astype(np.float32)

# ============================================== 02 RAYOS + 01 HALO (angular)
valid = (m_frame < 0.5) & (dl > 126) & (dl < 258)
bs = np.zeros(NA); bc = np.zeros(NA)
np.add.at(bs, ai[valid], lum[valid])
np.add.at(bc, ai[valid], 1.0)
good = bc > 0
am = np.interp(np.arange(NA), np.arange(NA)[good], (bs / np.maximum(bc, 1))[good])
am = cv2.GaussianBlur(am.reshape(1, -1), (0, 0), 2.0).ravel()
halo = cv2.GaussianBlur(am.reshape(1, -1), (0, 0), 26).ravel()
ray_prof = np.clip(am - halo, 0, None)
ray_prof = ray_prof / max(ray_prof.max(), 1e-6)
ray_prof = cv2.GaussianBlur(ray_prof.reshape(1, -1), (0, 0), 1.3).ravel()

rad_r = np.clip((dl - 124) / 20.0, 0, 1) * np.clip((258 - dl) / 118.0, 0, 1) ** 1.5
ray_a = blur(np.clip(ray_prof[ai] * rad_r, 0, 1) ** 0.8, 0.5)
ray_rgb = np.tile(np.array([1.00, 0.94, 0.80], np.float32), (H, W, 1))

halo_prof = np.clip(halo, 0, None) / max(np.clip(halo, 0, None).max(), 1e-6)
halo_rad = np.clip((268 - dl) / 150.0, 0, 1) ** 2.1
halo_a = blur(np.clip(halo_prof[ai] * halo_rad, 0, 1), 3.0) * 0.85
halo_rgb = np.tile(np.array([1.00, 0.86, 0.62], np.float32), (H, W, 1))

# ================================================================= 06 CHICA
cap = ((dl <= R - 2) & (lum < 0.80)).astype(np.float32)
cap = op(cap, 7, 1, cv2.MORPH_CLOSE)
cap = largest_cc(cap, near=(240, 250))

poly = np.array([(170, 300), (196, 296), (240, 300), (290, 296), (322, 300), (330, 330),
                 (330, 372), (322, 400), (320, 430), (316, 452), (330, 470), (348, 492),
                 (360, 508), (352, 522), (300, 532), (250, 538), (200, 532), (150, 522),
                 (142, 506), (152, 490), (168, 468), (182, 448), (178, 430), (174, 400),
                 (168, 372), (164, 340)], np.int32)
tm = np.full((H, W), 2, np.uint8)
pm = np.zeros((H, W), np.uint8)
cv2.fillPoly(pm, [poly], 1)
tm[pm > 0] = 1
out0 = np.zeros((H, W), bool)
out0[:128, :] = out0[548:, :] = True
out0[:, :134] = out0[:, 374:] = True
tm[out0] = 0
gc = tm.copy()
cv2.grabCut(cv2.GaussianBlur(RGB, (5, 5), 0), gc, None,
            np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
            8, cv2.GC_INIT_WITH_MASK)
body = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.float32)
body = op(body, 9, 1, cv2.MORPH_CLOSE)
body = op(body, 5, 1, cv2.MORPH_OPEN)

girl = np.clip(cap + body, 0, 1)
girl = op(girl, 9, 1, cv2.MORPH_CLOSE)
girl = largest_cc(girl, near=(240, 300))
girl = fill_small_holes(girl, 900)
# el relleno absorbio la punta del titulo: quitar solo el blanco puro del interior
tit_wh = (lum > 0.72) & (sat < 0.16)
tit_wh[:514, :] = False
girl = np.clip(girl - tit_wh, 0, 1)
girl = blur(girl, 0.6)

teal = (b > r + 0.04) & (g > r + 0.01) & (sat > 0.45) & (lum > 0.05)
dress = blur(op((teal & (girl > 0.4)).astype(np.float32), 3, 1, cv2.MORPH_CLOSE), 0.5)

# ================================================================= 05 MARIPOSAS
bf_col = (lum > 0.55) & (sat < 0.40)
zone = np.zeros((H, W), bool); zone[336:505, :] = True
bf = (bf_col & zone & (girl < 0.30) & (m_frame < 0.40) & (dl > 140)).astype(np.uint8)
bf = op(bf, 7, 1, cv2.MORPH_CLOSE)
bf = keep_cc(bf, 120, minw=10, minh=10, maxarea=3000)
bf = blur(bf, 0.5)

# ================================================================= 08 CHISPAS
free = (girl < 0.15) & (m_frame < 0.15) & (bf < 0.15)
lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
sp = ((lap > 28) & (lum > 0.45) & free & (sat < 0.55)).astype(np.uint8)
spark = keep_cc(sp, 3, drop_solid=False, maxarea=60)
spark[~(spark.astype(bool) & (np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) > 0))] = 0
n, lbl, st, _ = cv2.connectedComponentsWithStats(spark.astype(np.uint8), 8)
spark = np.zeros((H, W), np.float32)
for i in range(1, n):
    x, y, w, h, area = st[i]
    if 2 <= w <= 9 and 2 <= h <= 9 and 3 <= area <= 60:
        spark[lbl == i] = 1
spark = blur(spark, 0.4)

# ================================================================= TEXTOS
def txt(y0, y1, x0, x1, thr, satmax, th=None, k=7):
    """th=None -> luminancia directa (texto grande y blanco).
       th!=None -> top-hat (texto fino sobre fondo oscuro variable)."""
    m = np.zeros((H, W), np.float32)
    if th is None:
        cond = (lum > thr) & (sat < satmax)
    else:
        cond = (tophat(k) > th) & (sat < satmax)
    m[y0:y1, x0:x1] = cond[y0:y1, x0:x1]
    # CLOSE (no OPEN): une los trazos finos del texto sin perderlos
    return blur(op(m, 3, 1, cv2.MORPH_CLOSE), 0.4)

m_titulo = txt(514, 598, 72, 418, 0.72, 0.16)              # blanco grande
m_autora = txt(638, 676, 126, 364, 0, 0.22, th=18, k=7)   # serif fino
m_logo   = txt(694, 740, 194, 298, 0, 0.20, th=18, k=7)   # logo petit

Yr = np.arange(H)[:, None]; Xr = np.arange(W)[None, :]
gold = (sat > 0.34) & (r > b + 0.28) & (r > 0.30)
m_rombo = np.zeros((H, W), np.float32)
m_rombo[598:650, 148:352] = gold[598:650, 148:352]
m_rombo = blur(op(m_rombo, 3, 1, cv2.MORPH_CLOSE), 0.4)

# ================================================================= GUARDAR
white = np.ones((H, W, 3), np.float32)
bg = np.tile(np.array([0.012, 0.010, 0.010], np.float32), (H, W, 1))
rombo_rgb = np.tile(np.array([0.88, 0.52, 0.20], np.float32), (H, W, 1))

print("capas generadas:")
for nm, c, a in [
    ("00_fondo_negro",      bg,       np.ones((H, W), np.float32)),
    ("01_halo_resplandor",  halo_rgb, halo_a),
    ("02_rayos_luz",        ray_rgb,  ray_a),
    ("03_luna",             moon_rgb, moon_a),
    ("04_marco_ornamental", A,        m_frame),
    ("05_mariposas",        white,    bf),
    ("06_chica",            A,        girl),
    ("07_vestido_turquesa", A,        dress),
    ("08_chispas",          white,    spark),
    ("09_titulo_between",   white,    m_titulo),
    ("10_autora",           white,    m_autora),
    ("11_adorno_rombo",     rombo_rgb, m_rombo),
    ("12_logo_editorial",   white,    m_logo),
]:
    save(nm, c, a)

for nm, arr in [("girl", girl), ("frame", m_frame), ("dress", dress), ("bf", bf),
                ("ray", ray_a), ("halo", halo_a), ("moon", moon_a), ("spark", spark)]:
    np.save(f"_f_{nm}.npy", arr)
np.save("_f_rayprof.npy", ray_prof)
np.save("_f_titulo.npy", m_titulo)
np.save("_f_autora.npy", m_autora)
np.save("_f_logo.npy", m_logo)
np.save("_f_rombo.npy", m_rombo)
print("OK ->", OUT)
