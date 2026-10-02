"""Exporta la v2 (luna mejorada) junto a la v1, sin sobrescribir nada."""
import glob, os, subprocess
import imageio_ffmpeg

exe = imageio_ffmpeg.get_ffmpeg_exe()
FR = "anim2/frames"
files = sorted(glob.glob(f"{FR}/f*.png"))
N = len(files)
print("fotogramas:", N)


def run(args, label):
    r = subprocess.run([exe, "-y", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode:
        print(f"  ERROR {label}:", r.stderr[:400])
    else:
        print(f"  {label}: {os.path.getsize(args[-1])/1024:.0f} KB")


run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_v2.mp4"], "mp4 1080p v2")

run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=720:1280:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_web_v2.mp4"], "mp4 720p v2")

run(["-framerate", "30", "-i", f"{FR}/f%05d.png",
     "-vf", "scale=540:960:flags=lanczos",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "29", "-preset", "veryslow",
     "-movflags", "+faststart", "anim2/BETWEEN_buildup_story_v2.mp4"], "mp4 540p v2")

from PIL import Image
Image.open("anim2/poster.jpg").convert("RGB").save("anim2/poster_v2.jpg",
                                                   quality=92)
print("poster_v2.jpg listo")

print("\n--- comparativa v1 / v2 ---")
for f in sorted(glob.glob("anim2/*.mp4")) + ["anim2/poster.jpg", "anim2/poster_v2.jpg"]:
    if os.path.isfile(f):
        print("  %9.0f KB  %s" % (os.path.getsize(f) / 1024, f))
