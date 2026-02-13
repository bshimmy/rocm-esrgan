import os
import time
import subprocess

INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

print("Watcher started")

while True:
    try:
        files = sorted(
            f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(INPUT_DIR, f))
        )

        for f in files:
            input_path = os.path.join(INPUT_DIR, f)
            output_path = os.path.join(OUTPUT_DIR, f[:-4] + "_upscaled.mp4")

            if not os.path.exists(output_path):
                print(f"Processing {f}")
                subprocess.run([
                    "/opt/venv/bin/python",
                    "/app/upscale_video_once.py",
                    "--input", input_path,
                    "--output", output_path,
                    "--scale", "4",
                    "--crf", "16",
                    "--preset", "veryslow"
                ], check=True)

        time.sleep(10)

    except Exception as e:
        print("Watcher error:", e)
        time.sleep(10)
