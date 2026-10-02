"""
Portada animada 'build-up' vertical 1080x1920.
 - Luna centrada horizontalmente
 - Titulo compuesto en espacio oscuro limpio
 - Toda la escena se arma por etapas
Salida: anim2/ (fotogramas .png)
"""
import os
import numpy as np
import cv2
from PIL import Image

CW, CH = 1080, 1920
FPS, DUR = 30, 15.0
NFR = int(FPS * DUR)
S = 1.78
OX, OY = 243.0, 232.0            # centro de la luna en la capa original (495x760)
MX, MY = 540.0, 640.0            # centro de la luna en el lienzo
DX, DY = int(round(MX - OX * S)), int(round(MY - OY * S))
OUT = "anim2"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)
YY, XX = np.mgrid[0:CH, 0:CW].astype(np.float32)
dl = np.sqrt((XX - MX) ** 2 + (YY - MY) ** 2)
ang = np.arctan2(YY - MY, XX - MX)
NA = 1440
ai = np.mod(np.floor((ang + np.pi) / (2 * np.pi) * NA).astype(int), NA)

# ------------------------------------------------------------------ carga
def load(n):
    a = np.asarray(Image.open(f"capas/{n}.png").convert("RGBA")).astype(np.float32) / 255.0
    return a[..., :3].copy(), a[..., 3].copy()


def scaled(n):
    rgb, a = load(n)
    h, w = a.shape
    nh, nw = int(round(h * S)), int(round(w * S))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    return r, m


def tex(n):
    rgb, a = load(n)
    ys, xs = np.nonzero(a > 0.02)
    return rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], \
        a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


print("cargando capas...")
halo_r, halo_a = scaled("01_halo_resplandor")
rays_r, rays_a = scaled("02_rayos_luz")
moon_r, moon_a = scaled("03_luna")
frame_r, frame_a = scaled("04_marco_ornamental")
girl_r, girl_a = scaled("06_chica")
dress_r, dress_a = scaled("07_vestido_turquesa")
bf_r, bf_a = scaled("05_mariposas")
spark_r, spark_a = scaled("08_chispas")

T_tit, T_tita = tex("09_titulo_between")
T_aut, T_auta = tex("10_autora")
T_rom, T_roma = tex("11_adorno_rombo")
T_log, T_loga = tex("12_logo_editorial")

# ------------------------------- perfil angular de los rayos (medido en la capa)
def angular_profile():
    """Muestrea la luminancia en un anillo alrededor de la luna y separa
    el patron angular de los rayos del halo de baja frecuencia."""
    lum = np.asarray(Image.open("Between.png").convert("RGB")).astype(np.float32)
    lum = (0.299 * lum[..., 0] + 0.587 * lum[..., 1] + 0.114 * lum[..., 2]) / 255.0
    yy, xx = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
    d = np.sqrt((xx - OX) ** 2 + (yy - OY) ** 2)
    m = (d > 122) & (d < 250)
    aidx = np.mod(np.floor((np.arctan2(yy - OY, xx - OX) + np.pi) /
                           (2 * np.pi) * NA).astype(int), NA)
    bs = np.zeros(NA); bc = np.zeros(NA)
    np.add.at(bs, aidx[m], lum[m])
    np.add.at(bc, aidx[m], 1.0)
    g = bc > 0
    idx = np.arange(NA)
    prof = np.interp(idx, idx[g], (bs / np.maximum(bc, 1))[g])
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 2.0).ravel()
    halo = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 26).ravel()
    rays = np.clip(prof - halo, 0, None)
    rays /= max(rays.max(), 1e-6)
    rays = cv2.GaussianBlur(rays.reshape(1, -1), (0, 0), 1.3).ravel()
    halo_n = np.clip(halo, 0, None)
    halo_n /= max(halo_n.max(), 1e-6)
    return rays, halo_n


ray_prof, halo_prof = angular_profile()
print("perfil de rayos listo")

# ------------------------------- mariposas sueltas
n, lbl, st, _ = cv2.connectedComponentsWithStats((bf_a > 0.35).astype(np.uint8), 8)
butterflies = []
for i in range(1, n):
    x, y, w, h, area = st[i]
    if area < 40 or w < 6 or h < 6:
        continue
    pad = 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bf_a.shape[1], x + w + pad), min(bf_a.shape[0], y + h + pad)
    m = (lbl[y0:y1, x0:x1] == i).astype(np.float32)
    tint = np.array([1.0, 0.985, 0.945], np.float32)
    butterflies.append(dict(
        x=x0, y=y0, w=x1 - x0, h=y1 - y0, a=m,
        rgb=(tint[None, None, :] * m[..., None] * 0.95).astype(np.float32),
        cx=x0 + w / 2, cy=y0 + h / 2,
        ph=rng.uniform(0, 6.28), sp=rng.uniform(0.8, 1.3),
        amp=rng.uniform(0.6, 1.5) * 40, dx=rng.uniform(-70, 70), dy=rng.uniform(-60, 40),
        depth=rng.uniform(0.7, 1.4)))
print("mariposas sueltas:", len(butterflies))

# ------------------------------- chispas sueltas
n2, l2, s2, _ = cv2.connectedComponentsWithStats((spark_a > 0.5).astype(np.uint8), 8)
sparks = []
for i in range(1, n2):
    x, y, w, h, area = s2[i]
    if not (3 * S * S * 0.4 <= area <= 60 * S * S * 1.6):
        continue
    ys, xs = np.nonzero(l2 == i)
    col = 0.55 + 0.45 * rng.random(3)
    sparks.append(dict(x=float(xs.mean()), y=float(ys.mean()),
                       ph=rng.uniform(0, 6.28), sp=rng.uniform(0.6, 1.7),
                       r=rng.uniform(2.5, 5.0), col=col.astype(np.float32)))
print("chispas:", len(sparks))

# ------------------------------- colocacion de textos
def stamp(canvas, rgb, a, cx, cy, scale, alpha=1.0, dx=0.0):
    h, w = a.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    m = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if alpha < 1.0:
        m = m * alpha
    x0 = int(round(cx - nw / 2 + dx)); y0 = int(round(cy - nh / 2))
    ax0, ay0 = max(0, -x0), max(0, -y0)
    ax1, ay1 = min(nw, CW - x0), min(nh, CH - y0)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    r2, m2 = r[ay0:ay1, ax0:ax1], m[ay0:ay1, ax0:ax1]
    px0, py0 = x0 + ax0, y0 + ay0
    ph, pw = m2.shape
    px1, py1 = min(CW, px0 + pw), min(CH, py0 + ph)
    if px1 <= px0 or py1 <= py0:
        return
    r2, m2 = r2[:py1 - py0, :px1 - px0], m2[:py1 - py0, :px1 - px0]
    al = m2[..., None]
    canvas[py0:py1, px0:px1] = r2 * al + canvas[py0:py1, px0:px1] * (1 - al)


# el titulo se compone como barrido horizontal con borde luminoso
def stamp_wipe(canvas, rgb, a, cx, cy, scale, prog, alpha=1.0):
    """prog 0..1: solo se revela la parte izquierda del ancho."""
    h, w = a.shape
    nh, nw = int(round(h * scale)), int(round(w * scale))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_CUBIC) * alpha
    x0 = int(round(cx - nw / 2)); y0 = int(round(cy - nh / 2))
    cut = int(round(prog * nw))
    if cut <= 0:
        return
    m[:, cut:] = 0.0
    ax0, ay0 = max(0, -x0), max(0, -y0)
    ax1, ay1 = min(nw, CW - x0), min(nh, CH - y0)
    if ax1 <= ax0 or ay1 <= ay0:
        return
    r2, m2 = r[ay0:ay1, ax0:ax1], m[ay0:ay1, ax0:ax1]
    px0, py0 = x0 + ax0, y0 + ay0
    ph, pw = m2.shape
    px1, py1 = min(CW, px0 + pw), min(CH, py0 + ph)
    if px1 <= px0 or py1 <= py0:
        return
    r2, m2 = r2[:py1 - py0, :px1 - px0], m2[:py1 - py0, :px1 - px0]
    al = m2[..., None]
    canvas[py0:py1, px0:px1] = r2 * al + canvas[py0:py1, px0:px1] * (1 - al)


# -------------------------------------------------------------ayout de texto
frame_bottom = DY + int(689 * S)
girl_bottom = DY + int(548 * S)
TITLE_Y = 1600
ROM_Y = 1740
AUT_Y = 1800
LOGO_Y = 1875
T_SCALE = 2.30
print(f"lienzo: luna({MX:.0f},{MY:.0f}) marco hasta y={frame_bottom} "
      f"chica hasta y={girl_bottom} titulo centrado en y={TITLE_Y}")



# =============================================================== TIMELINE
# cada etapa: (inicio, fin) en segundos
ST = {
    "glow":     (0.00, 1.10),   # primera luz en el fondo
    "moon":     (0.55, 2.10),   # el disco aparece y crece
    "rays":     (1.35, 3.30),   # los rayos salen hacia fuera
    "halo":     (1.90, 3.60),   # el halo se abre
    "frame":    (2.70, 5.10),   # el marco dorado se dibuja de arriba abajo
    "figure":   (4.30, 6.40),   # la chica y el vestido emergen
    "butterfly": (5.70, 7.80),  # las mariposas entran volando
    "spark":    (6.90, 8.60),   # las chispas se encienden
    "title":    (7.80, 10.40),  # las letras se componen (barrido)
    "rombo":    (10.10, 11.10),
    "author":   (10.80, 11.90),
    "logo":     (11.50, 12.60),
}
HOLD_END = 14.10               # hasta aqui queda respirando
FADE = (14.10, 15.00)          # y se apaga para volver a empezar


# ---------------------------------------------- rayos anchos (profundidad)
_WIDE = cv2.GaussianBlur(ray_prof.reshape(1, -1), (0, 0), 9).ravel()


def wide_rays(t, p):
    """Segunda capa de luz: mas ancha, mas lenta y mas difusa que los
    rayos finos. Es lo que da profundidad al haz."""
    if t < ST["rays"][0]:
        return None
    prof = np.roll(_WIDE, int(-t * 3.0))
    rad = (np.clip((dl - R_MOON * 0.55) / 70.0, 0, 1) *
           np.clip((360.0 - dl) / 230.0, 0, 1) ** 1.3)
    a = np.clip(prof[ai] * rad, 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), 16)
    a *= (0.62 + 0.38 * np.sin(p * 2 + 2.0))
    return a * 0.34


# ---------------------------------------------- motas de polvo
def _make_motes():
    rng = np.random.default_rng(11)
    return [dict(x=rng.random() * CW, y=rng.random() * CH,
                 r=rng.random() * 1.9 + 1.0,
                 sp=rng.random() * 15.0 + 5.0,
                 ph=rng.uniform(0, 6.283),
                 drift=rng.uniform(-11, 11),
                 cool=rng.random() < 0.26) for _ in range(56)]


MOTES = _make_motes()


def draw_motes(img, t, p):
    """Polvo suspendido: solo se ve dentro del haz, por eso resulta creible."""
    for m in MOTES:
        y = (m["y"] - t * m["sp"]) % (CH + 90.0) - 45.0
        x = m["x"] + np.sin(p * m["sp"] * 0.11 + m["ph"]) * m["drift"]
        d = np.hypot(x - MX, y - MY)
        lit = np.clip(1.0 - d / (R_MOON * 3.4), 0, 1) ** 1.6
        tw = 0.5 + 0.5 * np.sin(p * 1.7 + m["ph"])
        a = lit * (0.14 + 0.52 * tw)
        if a < 0.012:
            continue
        col = (0.52, 0.92, 0.90) if m["cool"] else (1.0, 0.93, 0.78)
        r = int(m["r"] * 2) + 2
        cx, cy = int(x), int(y)
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(CW, cx + r + 1), min(CH, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        gy, gx = np.mgrid[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        g = np.exp(-(gx ** 2 + gy ** 2) / (2.0 * (m["r"] * 1.5) ** 2))
        img[y0:y1, x0:x1] += np.array(col, np.float32) * (g * a * 0.5)[..., None]
    return img


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)          # smoothstep


def prog(t, key, easeit=True):
    a, b = ST[key]
    u = (t - a) / max(b - a, 1e-6)
    u = min(1.0, max(0.0, u))
    return ease(u) if easeit else u


def stage_in(t, key, extra=0.0):
    """1 cuando la etapa ya empezo (paraipolar animaciones no acumulativas)."""
    return 1.0 if t >= ST[key][0] + extra else 0.0



def win(layer_alpha):
    """Ventana del lienzo que corresponde a una capa escalada."""
    h, w = layer_alpha.shape
    return slice(max(0, DY), min(CH, DY + h)), slice(max(0, DX), min(CW, DX + w))


def layer_mask(layer_alpha, canvas_mask, blur=0.6):
    """Mascara de lienzo recortada al tamano de la capa y suavizada."""
    sy, sx = win(layer_alpha)
    m = canvas_mask[sy, sx]
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return np.clip(m, 0, 1)


def comp(img, rgb, alpha, canvas_mask, blur=0.6, gain=1.0, boost=None):
    """Mezcla una capa escalada (rgb, alpha) sobre el lienzo.
    La mascara de animacion (canvas_mask) MODULA el alfa propio de la capa:
    los dos se multiplican, si no la capa se pintaria como un rectangulo opaco
    y arrastraria los pixeles invisibles que aun guarda en RGB.
    boost: mapa de luz de tamano lienzo que aclaro los pixeles de la capa."""
    sy, sx = win(alpha)
    a = canvas_mask[sy, sx] * alpha
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    a = np.clip(a, 0, 1) * gain
    hh = min(alpha.shape[0], sy.stop - sy.start)
    ww = min(alpha.shape[1], sx.stop - sx.start)
    if hh <= 0 or ww <= 0:
        return
    a = a[:hh, :ww][..., None]
    r = rgb[:hh, :ww]
    if boost is not None:
        r = np.clip(r * (1.0 + boost[sy.start:sy.start + hh,
                                      sx.start:sx.start + ww][..., None]), 0, 1)
    d = img[sy.start:sy.start + hh, sx.start:sx.start + ww]
    img[sy.start:sy.start + hh, sx.start:sx.start + ww] = r * a + d * (1 - a)

R_MOON = 127.0 * S             # radio de la luna en el lienzo


# ======================================================= LUNA MEJORADA
# Todo se precalcula a baja resolucion y se reescala: el coste por
# fotograma se mantiene bajo.
_SW = CW // 3
_SH = CH // 3
_sx, _sy = _SW / CW, _SH / CH


def _make_surface():
    """Maria lunares: ruido suave normalizado, con algo de craterizado."""
    rng = np.random.default_rng(3)
    n = _SW
    z = rng.random((n, n)).astype(np.float32)
    z = cv2.GaussianBlur(z, (0, 0), n * 0.045)
    z = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    z = cv2.GaussianBlur(z, (0, 0), n * 0.022)
    # normalizar alrededor de 0.5 para modular sin oscurecer de mas
    z = 0.5 + (z - 0.5) * 0.9
    return np.clip(z, 0, 1).astype(np.float32)


MOON_SURFACE = _make_surface()


def moon_surface_rot(t):
    """Gira la textura muy despacio (una vuelta cada ~95 s)."""
    deg = (t * 3.8) % 360.0
    M = cv2.getRotationMatrix2D((MX * _sx, MY * _sy), deg, 1.0)
    return cv2.warpAffine(MOON_SURFACE, M, (_SW, _SH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def moon_clouds(t):
    """Banda de nube suave cruzando el disco, en diagonal."""
    a = np.deg2rad(-24.0)
    ca, sa = np.cos(a), np.sin(a)
    proj = (XX - MX) * ca + (YY - MY) * sa          # coordenada de la banda
    span = R_MOON * 1.9
    pos = span * (0.62 * np.sin(2 * np.pi * t / 26.0) - 0.10)
    band = np.exp(-((proj - pos) ** 2) / (2 * (R_MOON * 0.42) ** 2))
    # segunda nube mas fina y lenta, para que no se lea como una sola
    pos2 = span * (0.55 * np.sin(2 * np.pi * t / 41.0 + 2.1))
    band += 0.55 * np.exp(-((proj - pos2) ** 2) / (2 * (R_MOON * 0.24) ** 2))
    return np.clip(band, 0, 1.35)


def moon_limb():
    """Anillo calido en el borde del disco."""
    w = R_MOON * 0.045
    return np.exp(-((dl - R_MOON) ** 2) / (2 * w * w))


def moon_ripples(t, mg):
    """Ondas concentricas que salen del disco mientras aparece."""
    out = np.zeros_like(dl)
    if mg <= 0 or mg >= 1:
        return out
    for k in range(3):
        ph = mg * 1.35 - k * 0.17
        if ph <= 0 or ph >= 1:
            continue
        rr = R_MOON * (0.92 + 0.62 * ph)
        w = R_MOON * (0.020 + 0.030 * ph)
        out += np.exp(-((dl - rr) ** 2) / (2 * w * w)) * (1 - ph) ** 1.4
    return out


def build_frame(t, first=False):
    p = 2 * np.pi * t
    img = np.zeros((CH, CW, 3), np.float32)
    img[:] = np.array([0.010, 0.009, 0.012], np.float32)

    # ---------- 1) resplandor inicial del fondo
    g = prog(t, "glow")
    if g > 0:
        amb = np.clip(1 - dl / 1500.0, 0, 1) ** 2.2 * g * 0.5
        img += np.array([0.10, 0.07, 0.04], np.float32)[None, None, :] * amb[..., None]

    # ---------- 2) luna: textura, nubes, limbo, ondas y respiracion
    mg = prog(t, "moon")
    if mg > 0:
        ease_m = ease(mg)
        # respiracion: dos armonicos, como antes pero algo mas perceptible
        breath = 1.0 + 0.020 * np.sin(p * 2) + 0.009 * np.sin(p * 3 + 1.1)
        sc = (0.72 + 0.28 * ease_m) * breath
        R = R_MOON * sc

        # viraje termico: la luna vira a calido al inspirar y a frio al espirar
        warm = 0.5 + 0.5 * np.sin(p * 2 + 0.6)
        # base por debajo de 1 para que el relieve y las nubes no se quemen
        body = np.array([0.94, 0.925 - 0.030 * (1 - warm), 0.885 - 0.070 * (1 - warm)],
                        np.float32)

        disc = np.clip((R - dl) / (2.4 * sc) + 0.5, 0, 1) * mg
        disc = cv2.GaussianBlur(disc, (0, 0), 0.6)

        # relieve lunar: las maria OSCURECEN mas de lo que aclaran, que es
        # como se lee de verdad una luna llena
        surf = cv2.resize(moon_surface_rot(t), (CW, CH), interpolation=cv2.INTER_LINEAR)
        relief = (0.62 + 0.46 * surf) * disc

        # nube: oscurece un punto del disco
        cloud = 1.0 - moon_clouds(t) * 0.26 * disc
        shade = relief * cloud

        # el limbo calido vive justo en el borde del disco
        limb = moon_limb() * disc * (0.55 + 0.45 * warm)

        reg = np.zeros((CH, CW), np.float32)
        hh, ww = moon_a.shape
        reg[max(0, DY):min(CH, DY + hh), max(0, DX):min(CW, DX + ww)] = 1.0

        lit = body[None, None, :] * shade[..., None] * disc[..., None]
        lit[..., 0] += limb * 0.30
        lit[..., 1] += limb * 0.20
        lit[..., 2] += limb * 0.09
        a = np.clip(disc * reg, 0, 1)
        img = img * (1 - a[..., None]) + np.clip(lit, 0, 1) * a[..., None]

        # ondas de nacimiento, por encima del disco
        rip = moon_ripples(t, mg) * np.clip(1 - mg * 0.55, 0, 1)
        img += np.array([1.00, 0.92, 0.76], np.float32)[None, None, :] * rip[..., None] * 0.42

    # ---------- 3) rayos: burst radial + rotacion
    rg = prog(t, "rays")
    if rg > 0:
        rot = np.roll(ray_prof, int(t * 7))
        reach = 250 * ease(rg)
        rad = np.clip((dl - R_MOON * 0.95) / 30.0, 0, 1) * \
              np.clip((reach - dl) / 90.0, 0, 1)
        ra = np.clip(rot[ai] * rad, 0, 1) ** 0.8
        ra = cv2.GaussianBlur(ra, (0, 0), 1.2)
        ra *= (0.72 + 0.32 * np.sin(p * 2 + 0.9))
        ra = np.clip(ra, 0, 1)
        img += np.array([1.0, 0.94, 0.80], np.float32)[None, None, :] * ra[..., None]

        wr = wide_rays(t, p)
        if wr is not None:
            img += np.array([1.00, 0.90, 0.72], np.float32)[None, None, :] * wr[..., None]

    # ---------- 4) halo
    hg = prog(t, "halo")
    if hg > 0:
        ha = np.clip(halo_prof[ai] * np.clip((1580 - dl) / 1300.0, 0, 1) ** 2.0, 0, 1)
        ha = cv2.GaussianBlur(ha, (0, 0), 26 * S) * 0.80 * hg
        ha *= (0.85 + 0.20 * np.sin(p * 2 + 0.4))
        img += np.array([1.0, 0.86, 0.62], np.float32)[None, None, :] * np.clip(ha, 0, 1)[..., None]

    # ---------- 5) marco: barrido vertical de arriba abajo
    fg = prog(t, "frame")
    if fg > 0:
        top = DY + 6 * S
        bot = DY + 689 * S
        edge = top + (bot - top) * ease(fg)
        yv = YY
        wipe = np.clip((edge - yv) / (110 * S) + 0.5, 0, 1)
        comp(img, frame_r, frame_a, wipe, 0.8, gain=1.0)
        # borde luminoso de la linea que dibuja
        sy, sx = win(frame_a)
        hh = max(0, min(frame_a.shape[0], sy.stop - sy.start))
        ww = max(0, min(frame_a.shape[1], sx.stop - sx.start))
        if hh > 0 and ww > 0:
            fmask = np.zeros((CH, CW), np.float32)
            fmask[sy.start:sy.start + hh, sx.start:sx.start + ww] = (frame_a > 0.02)[:hh, :ww]
            line = np.exp(-((yv - edge) ** 2) / (2 * (16 * S) ** 2)) * (1 - ease(fg)) * 0.9
            line = np.clip(line, 0, 1) * np.clip(1.0 - np.abs(XX - MX) / 700.0, 0, 1)
            a2 = np.clip(line * fmask, 0, 1)
            img += np.array([1.0, 0.85, 0.55], np.float32)[None, None, :] * a2[..., None]

    # ---------- 6) chica + vestido: emergen desde abajo con barrido
    xg = prog(t, "figure")
    if xg > 0:
        yv = YY
        top = DY + 148 * S
        bot = DY + 548 * S
        edge = bot + 60 - (bot + 60 - (top - 40)) * ease(xg)
        wipe = np.clip((yv - edge) / (80 * S) + 0.5, 0, 1)
        sweep = np.clip(1.0 - np.abs((yv - (DY + 400 * S)) / (150 * S)) -
                        (0.5 + 0.5 * np.sin(p)), 0, 1) ** 3
        comp(img, girl_r, girl_a, wipe, 0.6)
        comp(img, dress_r, dress_a, wipe, 0.6, boost=0.5 * sweep)

    # ---------- 7) mariposas
    for bf in butterflies:
        k = prog(t, "butterfly")
        if k <= 0:
            continue
        # cada una entra con retardo propio segun su indice
        d = (bf["cx"] / CW)
        kb = ease(min(1.0, max(0.0, (k - d * 0.45) / 0.55)))
        if kb <= 0.01:
            continue
        sy = 1.0 + 0.13 * np.sin(p * bf["sp"] * 4 + bf["ph"])
        sx = abs(np.cos(p * bf["sp"] * 2 + bf["ph"])) * 0.55 + 0.45
        nh, nw = bf["h"], max(2, int(bf["w"] * sx))
        aa = cv2.resize(bf["a"], (nw, nh), interpolation=cv2.INTER_AREA) * kb
        cc = cv2.resize(bf["rgb"], (nw, nh), interpolation=cv2.INTER_AREA)
        # entran desde fuera hacia su sitio
        x = int(bf["cx"] - nw / 2 + bf["dx"] * (1 - kb) +
                bf["amp"] * 2.0 * np.sin(p * bf["sp"] + bf["ph"]))
        y = int(bf["cy"] - nh / 2 + bf["dy"] * (1 - kb) +
                bf["amp"] * 1.6 * np.sin(p * bf["sp"] * 1.4 + bf["ph"] * 1.3))
        _over(img, cc, aa, x, y, 0.86 + 0.14 * np.sin(p * bf["sp"] * 2 + bf["ph"]))

    # ---------- 8) chispas
    sg = prog(t, "spark")
    if sg > 0:
        for i, s in enumerate(sparks):
            d = rng_phase[i]
            ks = ease(min(1.0, max(0.0, (sg - d * 0.6) / 0.4)))
            if ks <= 0.02:
                continue
            k = ks * (0.5 + 0.5 * np.sin(p * s["sp"] * 3 + s["ph"])) ** 2.2
            if k < 0.06:
                continue
            r = int(s["r"] * 4)
            yy2, xx2 = np.mgrid[-r:r + 1, -r:r + 1]
            gf = np.exp(-(xx2 ** 2 + yy2 ** 2) / (2.0 * (s["r"] * 1.6) ** 2))
            px_, py_ = int(round(s["x"])), int(round(s["y"]))
            X0, Y0 = max(0, px_ - r), max(0, py_ - r)
            X1, Y1 = min(CW, px_ + r + 1), min(CH, py_ + r + 1)
            if X1 <= X0 or Y1 <= Y0:
                continue
            gg = gf[Y0 - (py_ - r):Y1 - (py_ - r), X0 - (px_ - r):X1 - (px_ - r)]
            img[Y0:Y1, X0:X1] += s["col"][None, None, :] * (gg * k * 0.42)[..., None]

    # ---------- 9) titulo: barrido con borde dorado
    tg = prog(t, "title")
    if tg > 0:
        up = ease(min(1.0, tg * 1.6))
        stamp_wipe(img, T_tit, T_tita, MX, TITLE_Y + (1 - up) * 26, T_SCALE, tg, 1.0)
        # borde luminoso en el frente del barrido
        wpx = T_tita.shape[1] * T_SCALE
        fx = MX - wpx / 2 + wpx * tg
        if 0.02 < tg < 0.999:
            xx2 = XX - fx
            gl = np.exp(-(xx2 ** 2) / (2 * (13.0) ** 2)) * \
                 np.clip(1 - np.abs(YY - TITLE_Y) / (T_tita.shape[0] * T_SCALE * 0.62), 0, 1)
            gl = np.clip(gl, 0, 1) * 0.85
            img += np.array([1.0, 0.90, 0.68], np.float32)[None, None, :] * gl[..., None]

    # ---------- 10-12) remate
    if t >= ST["rombo"][0]:
        stamp(img, T_rom, T_roma, MX, ROM_Y, 1.70, prog(t, "rombo"))
    if t >= ST["author"][0]:
        stamp(img, T_aut, T_auta, MX, AUT_Y, 1.75, prog(t, "author"))
    if t >= ST["logo"][0]:
        stamp(img, T_log, T_loga, MX, LOGO_Y, 1.70, prog(t, "logo"))

    # ---------- polvo en la luz
    if t > ST["spark"][0] - 1.0:
        img = draw_motes(img, t, p)

    # ---------- post
    return post(img, t, p)


def _over(dst, rgb, a, x, y, opacity=1.0):
    h, w = a.shape
    X0, Y0 = max(0, x), max(0, y)
    X1, Y1 = min(CW, x + w), min(CH, y + h)
    if X0 >= X1 or Y0 >= Y1:
        return
    sa = a[Y0 - y:Y1 - y, X0 - x:X1 - x] * opacity
    sc = rgb[Y0 - y:Y1 - y, X0 - x:X1 - x]
    sa = sa[..., None]
    dst[Y0:Y1, X0:X1] = sc * sa + dst[Y0:Y1, X0:X1] * (1 - sa)


def post(img, t, p):
    img = np.clip(img, 0, None)
    # bloom barato: en baja resolucion y de vuelta.
    # Umbral alto y fuerza contenida: si no, el nucleo de la luna se
    # satura y se pierde el relieve lunar.
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    small = cv2.resize(np.clip(lum - 0.82, 0, None), (CW // 6, CH // 6),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 5)
    bloom = cv2.resize(small, (CW, CH), interpolation=cv2.INTER_LINEAR) * 1.45
    img += bloom[..., None] * np.array([1.0, 0.93, 0.80], np.float32)

    # vineta
    vy = (YY - CH / 2) / (CH / 2); vx = (XX - CW / 2) / (CW / 2)
    vig = np.clip(1 - 0.60 * (vx ** 2 + vy ** 2) ** 0.75, 0.30, 1.0)
    img *= vig[..., None]

    # grano
    img += rng.normal(0, 0.010, (CH, CW, 1)).astype(np.float32)

    img = np.clip(img, 0, 1.5) ** (1 / 1.05)
    img = (img - 0.5) * 1.04 + 0.5

    # fundido final para cerrar el bucle
    fa, fb = FADE
    if t >= fa:
        k = 1.0 - (t - fa) / (fb - fa)
        img *= k * k
    return np.clip(img, 0, 1)


# fase de entrada retardada de cada chispa (determinista)
rng_phase = np.linspace(0, 0.62, len(sparks))


# ---------- prueba: un fotograma en cada etapa clave
if __name__ == "__main__" and "--test" in os.sys.argv:
    import time
    shots = [0.9, 2.0, 3.2, 4.6, 6.2, 7.6, 9.0, 10.2, 11.4, 12.6, 13.5]
    tiles = []
    t0 = time.time()
    for tt in shots:
        f = build_frame(tt)
        tiles.append((f * 255 + 0.5).astype(np.uint8))
        print(f"  t={tt:5.2f}s  ({time.time()-t0:5.1f}s acumulado)")
    # contactos 4 columnas
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw, th = CW // 4, CH // 4
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, f in enumerate(tiles):
        s = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = s
    Image.fromarray(sheet).save("anim2/_test_stages.png")
    for tt in shots:
        Image.fromarray((build_frame(tt) * 255 + 0.5).astype(np.uint8)).convert("RGB").save(
            f"anim2/t{tt:04.1f}.jpg", quality=88)
    print("hoja de etapas ->", f"{time.time()-t0:.1f}s")

# ---------- render completo
if __name__ == "__main__" and "--render" in os.sys.argv:
    import time
    fr = os.path.join(OUT, "frames")
    os.makedirs(fr, exist_ok=True)
    t0 = time.time()
    for i in range(NFR):
        t = i / FPS
        f = build_frame(t)
        Image.fromarray((f * 255 + 0.5).astype(np.uint8)).save(
            os.path.join(fr, f"f{i:05d}.png"), optimize=False, compress_level=1)
        if i % 25 == 0:
            el = time.time() - t0
            eta = el / max(i, 1) * (NFR - i)
            print(f"  {i:4d}/{NFR}  t={t:5.2f}s  {el/60:4.1f}min  ETA {eta/60:4.1f}min",
                  flush=True)
    print(f"listo: {NFR} fotogramas en {(time.time()-t0)/60:.1f} min -> {fr}")

    # poster: el ultimo fotograma antes del fundido
    Image.fromarray((build_frame(13.4) * 255 + 0.5).astype(np.uint8)).convert("RGB")\
         .save(f"{OUT}/poster.jpg", quality=92)
    print("poster ->", f"{OUT}/poster.jpg")
