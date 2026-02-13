import os
import subprocess
import argparse
import time
import glob
import shutil
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm

VALID_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
}


def run_cmd(cmd, label):
    print(f"--- {label} ---")
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f"{label} time: {end - start:.2f}s")
    return end - start


def ffmpeg_file_arg(path):
    abs_path = os.path.abspath(path)
    return f"file:{abs_path}"


def upscale_frames(model_path, input_dir, output_dir, scale):
    os.makedirs(output_dir, exist_ok=True)

    session = ort.InferenceSession(
        model_path,
        providers=["MIGraphXExecutionProvider"]
    )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Total frames: {len(frames)}")

    start = time.time()

    for frame in tqdm(frames):
        img = cv2.imread(frame)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {frame}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        output = session.run(None, {"image": img})[0]

        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, os.path.basename(frame))
        cv2.imwrite(out_path, output)

    end = time.time()
    print(f"Upscale stage time: {end - start:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()

    if args.crf < 0 or args.crf > 51:
        raise ValueError("--crf must be between 0 and 51")
    if args.preset not in VALID_PRESETS:
        raise ValueError(f"--preset must be one of: {', '.join(sorted(VALID_PRESETS))}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    input_video = ffmpeg_file_arg(args.input)
    output_video = ffmpeg_file_arg(args.output)

    frames_dir = "/work/frames"
    upscaled_dir = "/work/upscaled"

    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(upscaled_dir, ignore_errors=True)

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(upscaled_dir, exist_ok=True)

    # 1. Extract frames
    extract_time = run_cmd(
        ["ffmpeg", "-y", "-i", input_video, f"{frames_dir}/frame_%06d.png"],
        "Extracting Frames"
    )

    # 2. Upscale
    print("--- Upscaling ---")
    upscale_frames(
        "/models/Real-ESRGAN-x4plus.onnx",
        frames_dir,
        upscaled_dir,
        args.scale
    )

    # 3. Re-encode
    encode_time = run_cmd(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "25",
            "-i",
            f"{upscaled_dir}/frame_%06d.png",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            output_video,
        ],
        "Encoding"
    )

    total_time = extract_time + encode_time
    print(f"\nTotal processing time (excluding upscale stage): {total_time:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
