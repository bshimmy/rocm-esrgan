import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    print("Initializing MIGraphX session...")
    compile_start = time.perf_counter()

    session = ort.InferenceSession(
        "/models/Real-ESRGAN-x4plus.onnx",
        providers=["MIGraphXExecutionProvider"]
    )

    compile_end = time.perf_counter()
    print(f"Model compile time: {compile_end - compile_start:.2f}s")

    input_name = session.get_inputs()[0].name
    files = sorted(os.listdir(args.input))
    total = len(files)

    start_time = time.perf_counter()

    for i, fname in enumerate(files, start=1):
        frame_start = time.perf_counter()

        path = os.path.join(args.input, fname)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        output = session.run(None, {input_name: img})[0]

        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output, fname), output)

        frame_end = time.perf_counter()

        elapsed = frame_end - start_time
        fps = i / elapsed
        remaining = total - i
        eta = remaining / fps if fps > 0 else 0

        print(
            f"[{i}/{total}] "
            f"Frame {frame_end-frame_start:.2f}s | "
            f"Avg FPS {fps:.2f} | "
            f"ETA {eta/60:.1f} min"
        )

    total_time = time.perf_counter() - start_time
    print("\nUpscale complete")
    print(f"Total time: {total_time:.2f}s")
    print(f"Effective FPS: {total/total_time:.2f}")

if __name__ == "__main__":
    main()
