#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
import sys
import os

import numpy as np
import tensorflow as tf

# local utils
from utils.io_utils import load_and_preprocess, postprocess_to_uint8, save_image
from utils.model_utils import load_model_with_groups_fix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF logs

def parse_args():
    p = argparse.ArgumentParser(description="SAI-GAN Inference: reconstruct masked faces.")
    p.add_argument("--model", required=True, help="Path to .h5 model checkpoint.")
    p.add_argument("--input_dir", required=True, help="Directory with masked images.")
    p.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    p.add_argument("--size", type=int, nargs=2, default=(256, 256), help="Resize (H W), default 256 256.")
    p.add_argument("--exts", nargs="+", default=[".png", ".jpg", ".jpeg"], help="Allowed image extensions.")
    p.add_argument("--suffix", default="", help="Filename suffix added before extension.")
    return p.parse_args()

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    H, W = args.size

    # collect files
    files = []
    for ext in args.exts:
        files.extend(glob.glob(str(input_dir / f"*{ext}")))
    files = [Path(f) for f in sorted(files)]
    if not files:
        print(f"[WARN] No images found in {input_dir} with extensions {args.exts}")
        sys.exit(0)

    # friendlier GPU memory behavior (optional)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    print(f"[INFO] Loading model from: {args.model}")
    model = load_model_with_groups_fix(args.model)
    print(f"[INFO] Found {len(files)} images. Writing to {output_dir}")

    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc="Reconstructing", unit="img")
    except Exception:
        iterator = files

    for fpath in iterator:
        try:
            x = load_and_preprocess(fpath, size=(H, W))  # (1,H,W,C), [-1,1]
            y = model.predict(x, verbose=0)              # (1,H,W,C), [-1,1]
            y_uint8 = postprocess_to_uint8(y[0])         # (H,W,C) in [0,255]
            out_name = fpath.stem + args.suffix + fpath.suffix
            save_image(y_uint8, output_dir / out_name)
        except Exception as e:
            print(f"[ERROR] Skipping {fpath.name}: {e}")

    print("[DONE] Inference complete.")

if __name__ == "__main__":
    main()
