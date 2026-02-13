import os
import time
import subprocess

INPUT_DIR = "/input"
OUTPUT_DIR = "/output"

print("Watcher started")

while True:
    try:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]

        for f in files:
            input_path = os.path.join(INPUT_DIR, f)
            output_path = os.path.join(OUTPUT_DIR, f.replace(".mp4", "_upscaled.mp4"))

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
                ])

        time.sleep(10)

    except Exception as e:
        print("Watcher error:", e)
        time.sleep(10)
