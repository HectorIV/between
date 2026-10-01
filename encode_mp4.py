import os, glob, subprocess
import imageio_ffmpeg

exe = imageio_ffmpeg.get_ffmpeg_exe()
print("ffmpeg:", exe)

def run(args):
    r = subprocess.run([exe, "-y", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("ERR:", r.stderr[:800])
    return r.returncode

# MP4 alta calidad (H.264, yuv420p). 495 es impar -> se escala a 496 para x264
rc = run(["-framerate", "24", "-i", "anim/frames/f%04d.png",
          "-vf", "scale=496:760:flags=lanczos",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
          "-preset", "slow", "-movflags", "+faststart",
          "anim/BETWEEN_animada.mp4"])
print("mp4 rc", rc, os.path.getsize("anim/BETWEEN_animada.mp4")//1024 if rc==0 else "", "KB")

# version vertical 1080x1920 para redes (escala manteniendo relacion 495x760)
rc2 = run(["-framerate", "24", "-i", "anim/frames/f%04d.png",
           "-vf", "scale=1080:1920:flags=lanczos",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-preset", "slow", "-movflags", "+faststart",
           "anim/BETWEEN_animada_1080x1920.mp4"])
print("mp4 vertical rc", rc2, os.path.getsize("anim/BETWEEN_animada_1080x1920.mp4")//1024 if rc2==0 else "", "KB")
