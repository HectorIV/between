"""Exporta la v3 (invierno + texto nitido) junto a v1 y v2."""
import glob, os, subprocess
from PIL import Image
import imageio_ffmpeg

os.chdir(r"C:\Users\Héctor\Desktop\proyectos\between")
exe = imageio_ffmpeg.get_ffmpeg_exe()
FR = "anim2/frames"
print("fotogramas:", len(glob.glob(f"{FR}/f*.png")))


def run(args, label):
    r = subprocess.run([exe, "-y", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode:
        print(f"  ERROR {label}:", r.stderr[:400])
    else:
        print(f"  {label}: {os.path.getsize(args[-1])/1024:.0f} KB")


# CRF 22 en web (antes 26): el titulo va sobre fondo liso y ahi se nota
run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=720:1280:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_web_v3.mp4"], "mp4 720p v3 (crf22)")

run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_v3.mp4"], "mp4 1080p v3")

run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=540:960:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_story_v3.mp4"], "mp4 540p v3")

Image.open("anim2/poster.jpg").convert("RGB").save("anim2/poster_v3.jpg", quality=92)
print("  poster_v3.jpg")

print("\n--- todas las versiones ---")
for f in sorted(glob.glob("anim2/*.mp4")) + sorted(glob.glob("anim2/poster*.jpg")):
    print("  %9.0f KB  %s" % (os.path.getsize(f) / 1024, f))
