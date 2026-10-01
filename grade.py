"""Variante 'lado oscuro' de la portada: gradacion fria, sin recortes.
Reutilizable para cualquier imagen RGBA de la portada."""
import numpy as np
from PIL import Image


def grade_dark(path_in, path_out, veil=(10, 20, 46), veil_a=0.30,
               expo=0.86, teal_boost=1.45, warm_cut=0.72, tint_lift=6):
    im = Image.open(path_in).convert("RGBA")
    a = np.asarray(im).astype(np.float32) / 255.0
    rgb, alpha = a[..., :3].copy(), a[..., 3:4]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299*r + 0.587*g + 0.114*b

    # 1) fria el marfil/luna: se apaga y vira a azul
    bright = np.clip((lum - 0.55) / 0.45, 0, 1)[..., None]
    cool = np.array([0.62, 0.74, 0.92], np.float32)
    rgb = rgb * (1 - bright) + (rgb * cool) * bright

    # 2) recorta los tonos dorados (marco, piel) para apagar el "calor"
    warm = np.clip((r - b) / 0.28, 0, 1) * np.clip(1 - lum / 0.5, 0, 1)
    rgb[..., 0] *= (1 - (1 - warm_cut) * warm)
    rgb[..., 1] *= (1 - (1 - (warm_cut + 0.12)) * warm)

    # 3) potencia el turquesa del vestido
    teal = ((b > r + 0.05) & (g > r + 0.02)).astype(np.float32)
    rgb[..., 1] = np.where(teal > 0, np.clip(rgb[..., 1] * teal_boost, 0, 1), rgb[..., 1])
    rgb[..., 2] = np.where(teal > 0, np.clip(rgb[..., 2] * (teal_boost * 0.92), 0, 1), rgb[..., 2])

    # 4) velo frio y exposicion  (veil en escala 0-255 -> normalizar)
    veil_n = np.array(veil, np.float32) / 255.0
    rgb = rgb * (1 - veil_a) + veil_n * veil_a
    rgb *= expo

    # 5) levanta ligeramente las sombras hacia el azul (look "noche")
    #    tint_lift va en escala 0-255 -> normalizar
    shadow = np.clip(1 - lum * 2.2, 0, 1)[..., None]
    lift = np.array([tint_lift, tint_lift + 4, tint_lift + 14], np.float32) / 255.0
    rgb += shadow * lift * 0.5

    # 6) contraste en S
    rgb = np.clip((rgb - 0.42) * 1.12 + 0.42, 0, 1)

    out = np.dstack([np.clip(rgb, 0, 1), alpha])
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(path_out)
    return path_out


if __name__ == "__main__":
    # 1) el lado oscuro de la portada animada
    grade_dark("anim/poster.png", "assets/cover_dark.png",
               veil=(8, 18, 44), veil_a=0.34, expo=0.84, teal_boost=1.50)

    # 2) la portada en positivo pero fria, para el bloque de "la academia"
    grade_dark("anim/poster.png", "assets/cover_cool.png",
               veil=(14, 22, 40), veil_a=0.16, expo=0.94, teal_boost=1.18,
               warm_cut=0.80, tint_lift=3)
    print("ok")
