"""
Empaqueta la landing en un unico .html autocontenido (todo en base64),
para poder mandarlo por correo sin carpeta ni rutas rotas.
Tambien genera la version .zip con la estructura de carpetas.
"""
import base64
import io
import os
import zipfile

SRC = "landing.html"
s = io.open(SRC, encoding="utf-8").read()


def b64(path, mime):
    d = base64.b64encode(io.open(path, "rb").read()).decode("ascii")
    return f"data:{mime};base64,{d}"


print("incrustando recursos...")

# fondo del hero
s = s.replace('src="assets/hero_bg.jpg"',
              f'src="{b64("assets/opt/hero_bg.webp", "image/webp")}"')
print("  hero_bg.webp")

# poster de los dos <video> (imagen de reserva mientras carga)
s = s.replace('poster="anim/poster.png"', 'poster="__POSTER__"')
s = s.replace("__POSTER__", b64("assets/opt/poster_sm.jpg", "image/jpeg"))
print("  poster_sm.jpg")

# logotipo (aparece dos veces)
logo = b64("assets/opt/logo.webp", "image/webp")
n = s.count('src="assets/logo.png"')
s = s.replace('src="assets/logo.png"', f'src="{logo}"')
print(f"  logo.webp x{n}")

# mariposas del hero (se inyectan por JS)
for i in (1, 2, 3):
    s = s.replace(f'"assets/mariposa{i}.png"',
                  f'"{b64(f"assets/opt/mariposa{i}.webp", "image/webp")}"')
print("  mariposa1-3.webp")

# el video, incrustado
s = s.replace('src="anim/BETWEEN_animada.mp4"',
              f'src="{b64("anim/opt_cover.mp4", "video/mp4")}"')
print("  opt_cover.mp4")

# og:image apunta a un archivo que ya no existe en un html suelto -> inline
s = s.replace('content="assets/cover_dark.png"',
              f'content="{b64("assets/opt/og_dark.jpg", "image/jpeg")}"')
print("  og_dark.jpg")

#titulo del enlace al wrap: ya no hay carpeta de assets
os.makedirs("dist", exist_ok=True)
OUT = "dist/BETWEEN_landing.html"
io.open(OUT, "w", encoding="utf-8", newline="").write(s)
sz = os.path.getsize(OUT) / 1024
print(f"\n{OUT}  ->  {sz:.0f} KB  ({sz/1024:.2f} MB)")
