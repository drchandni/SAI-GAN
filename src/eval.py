#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

from utils.metrics_utils import compute_all

def load_uint8(path: Path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC)  # (W,H)
    return np.array(img, dtype=np.uint8)

def main():
    p = argparse.ArgumentParser("Evaluate reconstruction quality metrics")
    p.add_argument("--gt_dir", required=True, help="Directory with ground-truth images")
    p.add_argument("--pred_dir", required=True, help="Directory with reconstructed images")
    p.add_argument("--size", type=int, nargs=2, default=None, help="Optional (H W) resize before evaluation")
    p.add_argument("--csv", default="results/metrics.csv", help="Where to save metrics CSV")
    p.add_argument("--ext", default=".jpg", help="Image extension to look for (e.g., .png, .jpg)")
    args = p.parse_args()

    gt_dir = Path(args.gt_dir)
    pr_dir = Path(args.pred_dir)
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(list(gt_dir.glob(f"*{args.ext}")))
    if not gt_files:
        print(f"[WARN] No GT images with extension {args.ext} in {gt_dir}")
        return

    rows, sums, count = [], {"PSNR":0,"SSIM":0,"UIQI":0,"NCORR":0,"MSE":0}, 0

    for gt_path in gt_files:
        pred_path = pr_dir / gt_path.name
        if not pred_path.exists():
            print(f"[SKIP] Missing prediction for {gt_path.name}")
            continue

        gt = load_uint8(gt_path, size=args.size)
        pr = load_uint8(pred_path, size=args.size)

        if gt.shape != pr.shape:  # safety
            h = min(gt.shape[0], pr.shape[0])
            w = min(gt.shape[1], pr.shape[1])
            gt = gt[:h,:w,:]
            pr = pr[:h,:w,:]

        m = compute_all(gt, pr)
        row = {"file": gt_path.name, **m}
        rows.append(row)
        for k in sums: sums[k] += m[k]
        count += 1

    if count == 0:
        print("[DONE] No valid pairs evaluated.")
        return

    avgs = {k: sums[k]/count for k in sums}

    # Write CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","PSNR","SSIM","UIQI","NCORR","MSE"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow({"file":"AVERAGES", **avgs})

    print(f"[DONE] Evaluated {count} pairs. CSV saved to {out_csv}")
    for k,v in avgs.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
