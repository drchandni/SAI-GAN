# SAI-GAN: Self-Attention Inpainting GAN for Masked Face Reconstruction
This repository showcases the **inference** and **evaluation** pipeline for **SAI-GAN**, a Pix2Pix-style conditional GAN that augments a **U-Net generator with self-attention** and uses a **PatchGAN discriminator**. The goal is to reconstruct masked facial regions while preserving local texture and overall facial coherence.  
This repo focuses on **running reconstruction on sample masked images** and computing image-quality metrics (**PSNR, SSIM, UIQI, NCORR, MSE**). Training code and any classifier-based validation are **out of scope** here and not included.

| Masked | Reconstructed |
|---|---|
| <img src="docs/examples/001_masked.jpg" width="220"> | <img src="docs/examples/001_recon.jpg" width="220"> |


## Features
- Self-Attention U-Net generator + PatchGAN discriminator  
- Clean, modular code in `src/utils` (I/O, model loading, metrics)  
- Inference script for masked inputs  
- Evaluation script with CSV export of PSNR/SSIM/UIQI/NCORR/MSE  
- Reproducible setup via `requirements.txt`

## Repository Structure
```
SAI-GAN/
├─ checkpoints/ # place .h5 model checkpoints here (gitignored)
├─ data/
│ ├─ masked/ # masked test images
│ └─ gt/ # ground-truth images (same filenames as recon)
├─ results/
│ └─ recon/ # reconstructed outputs get saved here
├─ src/
│ ├─ test.py # inference
│ ├─ eval.py # metrics computation
│ └─ utils/
│ ├─ io_utils.py
│ ├─ model_utils.py # Conv2DTranspose(groups) compatibility fix
│ └─ metrics_utils.py # PSNR/SSIM/UIQI/NCORR/MSE
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Setup

> **Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Key dependencies: tensorflow (2.10–2.16), numpy, pillow, tqdm, opencv-python, scikit-image.

**Quick Start**
1) Run Reconstruction
Place your trained checkpoint (e.g., checkpoints/model_038340.h5) and masked test images (data/masked/).
2) Then:
Windows
```powershell
python src\test.py --model checkpoints\model_038340.h5 --input_dir data\masked --output_dir results\recon --size 256 256 --suffix _recon
```
macOS / Linux
```bash
python src/test.py --model checkpoints/model_038340.h5 --input_dir data/masked --output_dir results/recon --size 256 256 --suffix _recon
```
Tip: If you want predictions to have exactly the same filenames as GT, run with --suffix "".

3) Evaluate Metrics
Ground-truth images go in data/gt/ with the same filenames as predictions. The evaluator is suffix-aware by default (_recon).
Windows
```powershell
python src\eval.py --gt_dir data\gt --pred_dir results\recon --size 256 256 --csv results\metrics.csv --ext .jpg --pred_suffix _recon --pred_ext .jpg
```

macOS / Linux
```bash
python src/eval.py --gt_dir data/gt --pred_dir results/recon --size 256 256 --csv results/metrics.csv --ext .jpg --pred_suffix _recon --pred_ext .jpg
```
Outputs:
Console: average PSNR, SSIM, UIQI, NCORR, MSE
results/metrics.csv: per-image metrics + an AVERAGES row

**Command Reference**

```
test.py (inference)
--model        Path to .h5 checkpoint (required)
--input_dir    Directory with masked images (required)
--output_dir   Directory for reconstructed outputs (required)
--size         Resize (H W), default 256 256
--exts         Allowed input extensions, default: .png .jpg .jpeg
--suffix       Suffix before extension for outputs (default: _recon). Use "" for none.
```

```
eval.py (metrics)
--gt_dir       Ground-truth directory (required)
--pred_dir     Predictions directory (required)
--size         Optional resize (H W) before evaluation
--csv          Output CSV path (default: results/metrics.csv)
--ext          GT image extension (e.g., .png or .jpg)
--pred_ext     Prediction extension (defaults to --ext)
--pred_suffix  Prediction filename suffix (default: _recon; set "" if none)
```

**Datasets & Checkpoints**
Datasets: Prepare masked/GT splits (e.g., CelebA-HQ variants). Respect original dataset licenses; this repo does not redistribute datasets.
Checkpoints: Place trained .h5 in checkpoints/. The loader handles TF/Keras versions without Conv2DTranspose(groups) by patching automatically.
Dataset for Training can be downloaded from the drive link - https://drive.google.com/drive/folders/1EJbxfgTVHDBNvfe7KzESwJoWc8e4J2HJ?usp=sharing 

**License**
Code: Licensed under Apache-2.0 (see LICENSE).
Documentation & images in /docs: CC BY 4.0.
Model weights: Apache-2.0 by default. If you prefer restricted use, update your MODEL_CARD.md and weights license accordingly.

**Citation**
If this project helps your research, please cite:
@article{agarwal2025saigan,
  title   = {SAI-GAN: Self-Attention Inpainting GAN for Masked Face Reconstruction},
  author  = {Agarwal, Chandni and Others},
  journal = {TBD},
  year    = {2025},
  note    = {Add DOI/arXiv when available}
}

**Contact**
Chandni Agarwal
Email: chandni1972@gmail.com 

**Acknowledgements**
Baselines: Pix2Pix, EdgeConnect, Gated Convolution, HVQVAE.
Frameworks: TensorFlow/Keras, NumPy, OpenCV, scikit-image.
