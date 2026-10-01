"""
Portada animada de "BETWEEN" a partir de las capas extraídas.
Efectos: luna que respira, rayos rotativos, halo pulsante, mariposas
aleteando, chispas titilando, viñeta y grano cinematográfico.
Salidas: Between_animada.webp (APNG-like), .gif, y un .mp4 si hay encoder.
"""
import os
import numpy as np
import cv2
from PIL import Image

W, H = 495, 760
FPS = 24
SECONDS = 6
NFR = FPS * SECONDS
OUT = "anim"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

CX, CY = 243.0, 232.0

L = {}
for f in sorted(os.listdir("capas")):
    if f.endswith(".png") and not f.startswith("_"):
        L[f[:-4]] = Image.open(os.path.join("capas", f)).convert("RGBA")
print("capas cargadas:", len(L))
SRC_RGB = np.asarray(Image.open("Between.png").convert("RGB"))

m_ray = np.load("_f_ray.npy")
ray_prof = np.load("_f_rayprof.npy")
NA = len(ray_prof)
YY, XX = np.mgrid[0:H, 0:W]
dl = np.sqrt((XX - CX) ** 2 + (YY - CY) ** 2)
ang = np.arctan2(YY - CY, XX - CX)
ai = np.mod(np.floor((ang + np.pi) / (2 * np.pi) * NA).astype(int), NA)

m_bf = np.load("_f_bf.npy")
m_spark = np.load("_f_spark.npy")
m_frame = np.load("_f_frame.npy")
m_halo = np.load("_f_halo.npy")
m_girl = np.load("_f_girl.npy")

# ---------------------------------------------------------------- butterflies
n, lbl, st, ct = cv2.connectedComponentsWithStats(m_bf.astype(np.uint8), 8)
bf_parts = []
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 120:
        continue
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    sub = (lbl[y0:y1, x0:x1] == i)
    rgb = np.ones((y1 - y0, x1 - x0, 3), np.float32)
    a = np.where(sub, 1.0, 0.0)
    # las alas son marfil: conserva el sombreado pero sube el base para que brillen
    src = SRC_RGB[y0:y1, x0:x1] / 255.0
    lum_src = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
    k = np.clip(0.55 + 1.35 * (lum_src - 0.48), 0.5, 1.12)
    tint = np.array([1.0, 0.985, 0.945], np.float32)
    rgb = (tint[None, None, :] * k[..., None]).astype(np.float32)
    bf_parts.append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, rgb=rgb, a=a,
                         cx=x + w / 2, cy=y + h / 2, ph=rng.uniform(0, 2 * np.pi),
                         sp=rng.uniform(0.7, 1.3), amp=rng.uniform(2.5, 7.0),
                         depth=rng.uniform(0.6, 1.4)))
print("mariposas individuales:", len(bf_parts))

# ---------------------------------------------------------------- chispas
n2, l2, s2, _ = cv2.connectedComponentsWithStats((m_spark > 0.5).astype(np.uint8), 8)
SRC_RGB = np.asarray(Image.open("Between.png").convert("RGB"))
sparks = []
for i in range(1, n2):
    x, y, w, h, area = s2[i]
    if not (3 <= area <= 60):
        continue
    ys, xs = np.nonzero(l2 == i)
    mx_, my_ = xs.mean(), ys.mean()
    base = SRC_RGB[ys, xs].mean(0) / 255.0
    # normalizar hacia un blanco-crema luminoso (las chispas son luz)
    base = 0.45 + 0.55 * np.clip(base / max(base.max(), 1e-3), 0, 1)
    sparks.append(dict(x=float(mx_), y=float(my_), ph=rng.uniform(0, 2 * np.pi),
                       sp=rng.uniform(0.5, 1.6), r=rng.uniform(1.6, 3.4),
                       col=base.astype(np.float32)))
print("chispas:", len(sparks))

SPK_GLOW = True


def over(dst, src_rgb, src_a, x, y, opacity=1.0):
    """Alpha-over 'src' onto float RGB 'dst' (H,W,3) at integer offset."""
    h, w = src_a.shape
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(W, x + w), min(H, y + h)
    if X0 >= X1 or Y0 >= Y1:
        return
    sa = src_a[Y0 - y:Y1 - y, X0 - x:X1 - x] * opacity
    sc = src_rgb[Y0 - y:Y1 - y, X0 - x:X1 - x]
    sa = sa[..., None]
    d = dst[Y0:Y1, X0:X1]
    dst[Y0:Y1, X0:X1] = sc * sa + d * (1 - sa)


# ---------------------------------------------------------------- efectos
def frame(idx, t):
    """t in [0,1) -> float RGB image"""
    p = t * 2 * np.pi
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array([0.012, 0.010, 0.010], np.float32)

    # --- luna: respiracion lenta + latido sutil
    breath = 1.0 + 0.022 * np.sin(p * 2) + 0.010 * np.sin(p * 3 + 1.1)
    moon = L["03_luna"]
    ma = np.asarray(moon)[..., 3] / 255.0
    mc = np.asarray(moon)[..., :3] / 255.0
    ma_s = np.clip(ma * breath, 0, 1)
    # segundo anillo de luz (aura de la luna)
    aura = np.clip(1.0 - (dl - R0) / 26.0, 0, 1) if False else None

    # --- halo pulsante
    halo_c = np.array([1.00, 0.86, 0.62], np.float32)
    ha = np.clip(m_halo * (1.0 + 0.16 * np.sin(p * 2 + 0.4)) * 0.92, 0, 1)
    img += halo_c[None, None, :] * ha[..., None]

    # --- rayos: rotacion lenta + respiracion de intensidad
    rot = np.roll(ray_prof, int(t * 8))          # 8 pasos de rotacion por vuelta
    rad = np.clip((dl - 124) / 20.0, 0, 1) * np.clip((258 - dl) / 118.0, 0, 1) ** 1.5
    ra = np.clip(rot[ai] * rad, 0, 1) ** 0.8
    ra = cv2.GaussianBlur(ra, (0, 0), 0.5)
    ra *= (0.78 + 0.30 * np.sin(p * 2 + 0.9))
    ra = np.clip(ra, 0, 1)
    img += np.array([1.00, 0.94, 0.80], np.float32)[None, None, :] * ra[..., None]

    # --- luna encima
    img = img * (1 - ma_s[..., None]) + mc * ma_s[..., None]

    # --- marco (parpadeo sutil de las chispas doradas)
    fr = L["04_marco_ornamental"]
    fa = np.asarray(fr)[..., 3] / 255.0
    fc = np.asarray(fr)[..., :3] / 255.0
    tw = 1.0 + 0.05 * np.sin(p * 2 + 1.7)
    img = img * (1 - fa[..., None]) + np.clip(fc * tw, 0, 1) * fa[..., None]

    # --- chica
    g = L["06_chica"]
    ga = np.asarray(g)[..., 3] / 255.0
    gc = np.asarray(g)[..., :3] / 255.0
    img = img * (1 - ga[..., None]) + gc * ga[..., None]

    # --- vestido turquesa: destello turquesa que recorre el vestido
    d = L["07_vestido_turquesa"]
    da = np.asarray(d)[..., 3] / 255.0
    dc = np.asarray(d)[..., :3] / 255.0
    sweep = np.clip(1.0 - np.abs(((YY - 340) / 150.0) - (0.5 + 0.5 * np.sin(p))), 0, 1) ** 3
    boost = 1.0 + 0.55 * sweep
    dc2 = np.clip(dc * boost[..., None], 0, 1)
    dc2[..., 2] = np.clip(dc2[..., 2] * 1.15, 0, 1)
    img = img * (1 - da[..., None]) + dc2 * da[..., None]

    # --- mariposas: aleteo (escala X) + flotacion
    for bf in bf_parts:
        sy = 1.0 + 0.13 * np.sin(p * bf["sp"] * 4 + bf["ph"])
        sx = abs(np.cos(p * bf["sp"] * 2 + bf["ph"])) * 0.55 + 0.45
        nh, nw = bf["h"], max(2, int(bf["w"] * sx))
        aa = cv2.resize(bf["a"], (nw, nh), interpolation=cv2.INTER_AREA)
        cc = cv2.resize(bf["rgb"], (nw, nh), interpolation=cv2.INTER_AREA)
        oy = bf["amp"] * np.sin(p * bf["sp"] + bf["ph"])
        ox = bf["amp"] * 0.6 * np.sin(p * bf["sp"] * 0.7 + bf["ph"] * 1.3)
        cx = int(round(bf["cx"] - nw / 2 + ox))
        cy = int(round(bf["cy"] - nh / 2 + oy))
        op_ = 0.86 + 0.14 * np.sin(p * bf["sp"] * 2 + bf["ph"])
        over(img, cc, aa, cx, cy, op_)

    # --- chispas: titileo
    for s in sparks:
        k = 0.5 + 0.5 * np.sin(p * s["sp"] * 3 + s["ph"])
        k = k ** 2.2
        if k < 0.08:
            continue
        if SPK_GLOW:
            r = int(s["r"] * 4)
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            gfall = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (s["r"] * 1.5) ** 2))
            px_, py_ = int(round(s["x"])), int(round(s["y"]))
            X0, Y0 = max(0, px_ - r), max(0, py_ - r)
            X1, Y1 = min(W, px_ + r + 1), min(H, py_ + r + 1)
            gg = gfall[Y0 - (py_ - r):Y1 - (py_ - r), X0 - (px_ - r):X1 - (px_ - r)]
            img[Y0:Y1, X0:X1] += s["col"][None, None, :] * (gg * k * 0.30)[..., None]
        # nucleo
        r2 = max(1, int(round(s["r"] * (0.5 + 0.4 * k))))
        px_, py_ = int(round(s["x"])), int(round(s["y"]))
        for dy in range(-r2, r2 + 1):
            for dx in range(-r2, r2 + 1):
                xx, yy = px_ + dx, py_ + dy
                if 0 <= xx < W and 0 <= yy < H:
                    f = 1.0 - (dx * dx + dy * dy) / (r2 * r2 + 1.0)
                    img[yy, xx] += s["col"] * (f * k * 0.80)

    # --- titulo, autora, logo, rombo
    for nm, opa in [("09_titulo_between", 1.0), ("10_autora", 1.0),
                    ("11_adorno_rombo", 1.0), ("12_logo_editorial", 1.0)]:
        lay = L[nm]
        la = np.asarray(lay)[..., 3] / 255.0
        lc = np.asarray(lay)[..., :3] / 255.0
        img = img * (1 - la[..., None]) + lc * la[..., None]

    # --- post: bloom en la luna, viñeta, grano, ligero tinte
    img = np.clip(img, 0, None)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    bloom = cv2.GaussianBlur(np.clip(lum - 0.72, 0, None), (0, 0), 14) * 1.5
    img += bloom[..., None] * np.array([1.0, 0.93, 0.80], np.float32)

    # viñeta
    vy = (YY - H / 2) / (H / 2)
    vx = (XX - W / 2) / (W / 2)
    vig = np.clip(1.0 - 0.55 * (vx ** 2 + vy ** 2) ** 0.75, 0.35, 1.0)
    img *= vig[..., None]

    # grano
    nse = rng.normal(0, 0.011, (H, W, 1)).astype(np.float32)
    img += nse * (0.35 + 0.65 * (1 - lum[..., None]))

    # gamma + contraste suave
    img = np.clip(img, 0, 1.6) ** (1 / 1.06)
    img = (img - 0.5) * 1.045 + 0.5
    return np.clip(img, 0, 1)


print("renderizando", NFR, "frames...")
frames = []
for i in range(NFR):
    f = frame(i, i / NFR)
    frames.append((f * 255 + 0.5).astype(np.uint8))
    if i % 48 == 0:
        print("  frame", i)

# guardar fotogramas sueltos
fd = os.path.join(OUT, "frames")
os.makedirs(fd, exist_ok=True)
for i, f in enumerate(frames):
    Image.fromarray(f).save(os.path.join(fd, f"f{i:04d}.png"))
print("frames guardados en", fd)
np.save("_anim_frames.npy", np.stack(frames))
print("OK")
