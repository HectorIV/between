"""Empaqueta los fotogramas de anim2 en MP4 / WebP / GIF."""
import glob, os, subprocess, sys
import imageio_ffmpeg
import numpy as np
from PIL import Image

exe = imageio_ffmpeg.get_ffmpeg_exe()
FR = "anim2/frames"
files = sorted(glob.glob(f"{FR}/f*.png"))
if not files:
    sys.exit("no hay fotogramas: ejecuta build_v2.py --render")
N = len(files)
print("fotogramas:", N)


def run(args, label):
    r = subprocess.run([exe, "-y", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode:
        print(f"  ERROR {label}:", r.stderr[:500])
    else:
        print(f"  ok {label}: {os.path.getsize(args[-1])/1024:.0f} KB")
    return r.returncode == 0


# --- MP4 alta calidad (1080x1920) ---
run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup.mp4"], "mp4hq")

# --- MP4 ligero para web/autoplay ---
run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=720:1280:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_web.mp4"], "mp4web")

# --- MP4 muy ligero (WhatsApp / stories) ---
run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=540:960:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "29", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_story.mp4"], "mp4story")

# --- WebP animado ---
ims = [Image.open(f).convert("RGB") for f in files[::2]]          # 15 fps
ims[0].save("anim2/BETWEEN_buildup.webp", save_all=True, append_images=ims[1:],
            duration=int(1000 / 15), loop=0, quality=82, method=4)
print(f"  ok webp: {os.path.getsize('anim2/BETWEEN_buildup.webp')/1024:.0f} KB  ({len(ims)} f)")

print("\nlisto.")
for f in sorted(glob.glob("anim2/*")):
    if os.path.isfile(f):
        print(f"  {os.path.getsize(f)/1024:9.0f} KB  {f}")
