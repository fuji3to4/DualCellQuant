# 🔬 DualCellQuant

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18347379.svg)](https://doi.org/10.5281/zenodo.18347379)

A web-based tool for **objective and reproducible quantification** of fluorescence intensity in the plasma membrane region. DualCellQuant integrates **Cellpose-SAM** automated segmentation with **Euclidean distance transform (EDT)–based radial normalization** to define membrane regions at a fixed normalized distance from the cell boundary, enabling accurate per-cell quantification without manual selection bias.

Two workflows are available via tabs in `dualCellQuant.py`:

- **Dual images**: Target + Reference images → AND mask → ratios and stats
- **Single image**: One image → (optional) Radial mask → Mask → stats

website:[DualCellQuant](https://dna00.bio.kyutech.ac.jp/dualcellquant/)

## Features ✨

- 🧠 **Cellpose-SAM automated segmentation**: Robust cell boundary detection (lazy model load, optional GPU)
- 🌀 **EDT-based radial normalization**: Shape-aware membrane region definition at normalized distances from cell boundaries (0% = center, 100% = boundary, >100% = expansion into background)
- 🎯 **Flexible masking strategies**: Reference-guided membrane region definition for robust quantification even with low-intensity target signals
  - none / global percentile / global Otsu / per-cell percentile / per-cell Otsu
  - Optional restriction to the radial ROI (per label)
- 📊 **Per-cell statistics**: AND(Target, Reference[, Radial]) → Mean/Sum per mask and per whole cell → Target/Reference ratio image and CSV export
- 🖼️ Clear overlays with cell IDs and downloadable NumPy arrays for all masks
- 🧹 Optional Preprocess: Background correction (Rolling ball) and Normalization (z-score, robust z, min-max, percentile)
  - Default: OFF

## Installation 📥

- Prereqs: Python 3.11+; optional GPU for Cellpose-SAM; install PyTorch appropriate for your CUDA/CPU setup first (see https://pytorch.org/get-started/locally/)
- Poetry
  - `poetry install`
  - `poetry run python dualCellQuant.py`
- Pip
  - `pip install .`
  - `python dualCellQuant.py`
- Pip (direct from GitHub)
  - `pip install "git+https://github.com/fuji3to4/DualCellQuant.git"`
- Optional: mount under FastAPI at `/dualcellquant`: `poetry run uvicorn serve:app --port 7860`

Then open the local Gradio URL shown in the terminal.

### Troubleshooting: CUDA issues

If CUDA is not detected or PyTorch is not using GPU:

1. Check CUDA version: `nvidia-smi`
2. Reinstall torch/torchvision matching your CUDA version (replace `cu130` with your version, e.g., `cu126`, `cu128`):

```pwsh
poetry run pip uninstall -y torch torchvision
poetry run pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
```

3. Verify installation:

```pwsh
poetry run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `True`, GPU support is enabled.


## Workflows ▶️

### Single image (new)

1. Run Cellpose-SAM segmentation on the uploaded image
2. Build Radial mask (optional)
3. Apply Mask (none/global/per-cell; saturation limit; min object size)
4. Quantify per cell (on-mask and whole-cell stats; CSV export)

Outputs: label mask TIFF, mask TIFF, table CSV, image overlay previews.

### Dual images

1. 🧩 Run Cellpose-SAM segmentation

- (Optional) Preprocess
  - Background correction (Rolling ball, radius in px)
  - Normalization: z-score (default), robust z-score (median/MAD), min-max, percentile [1,99]
  - Preview and download preprocessed images (8-bit TIFF)


- Choose source image/channel and model thresholds
- Outputs: label mask (.npy), overlays

### 2. 🌀 (Optional) Build Radial mask (EDT-based)

- **EDT-based radial normalization**: Per-cell Euclidean distance transform (EDT) normalized to [0..1] defines shape-aware ring between Inner% (0=center) and Outer% (100=boundary)
- Outer >100% expands outward into background only; labels are propagated via nearest cell EDT
- Outputs: radial mask (bool .npy), radial labels (.npy), overlay

### 3. 🎯 Apply Target mask

- Choose channel, saturation limit, masking mode, percentile (if applicable), min object size
- Optionally restrict to the Radial ROI (per-label)
- Outputs: target mask (.npy), overlay
  - Note: Saturation limit thresholding is computed on the original image intensities (0–1 scale) so that preprocessing does not affect saturation gating.

### 4. 📎 Apply Reference mask

- Same options as Target; can also use Radial ROI
- Outputs: reference mask (.npy), overlay

### 5. 📊 Integrate & Quantify

- Optionally AND with Radial mask
- Produces per-cell table (CSV), AND mask (.npy), and a Target/Reference ratio image (.npy and preview)
  - If Preprocess is enabled, both visualization and measurement use the preprocessed arrays (for Dual/Single). For saturation gating in masking, the original image scale is used.
  - Ratio (T/R) is computed using only pixels where Reference > 0. This avoids invalid divisions especially when Radial outer > 100% includes background outside cells.

## UI Controls (by section) 🧰

- Segmentation params
  - Segment on: target/reference
  - Segmentation channel: 0 (gray/RGB→gray), 1 (R), 2 (G), 3 (B)
  - Diameter (px, 0=auto), Flow threshold, Cellprob threshold, Use GPU
- Radial mask (optional)
  - Radial inner % (0=center), Radial outer % (100=boundary, >100 expands outward)
  - Remove small objects (px)
  - Outputs: radial mask (bool), radial labels
  - Checkboxes: Use Radial ROI for Target/Reference mask
- Target mask
  - Channel, Masking mode, Saturation limit, Percentile, Remove small objects
- Reference mask
  - Same as Target
- Integrate
  - AND with radial mask (toggle)

- Preprocess (optional)
  - Background correction (Rolling ball): radius in px
  - Normalization method: z-score / robust z-score / min-max / percentile [1,99]
  - Preview buttons and TIFF download of preprocessed images

## Key Implementation Notes 🛠️

- File: `dualCellQuant.py`
- Model: `cellpose.models.CellposeModel(pretrained_model="cpsam")`
- Image handling: PIL↔NumPy helpers, robust grayscale extraction
- Masking modes:
  - Global percentile/Otsu: threshold computed on non-saturated pixels of the whole image
  - Per-cell percentile/Otsu: threshold computed from pixels inside each cell (and non-saturated)
  - Cleanup: `remove_small_objects` and a small `binary_opening`
- Radial mask internals:
  - Inside (≤100%): per-cell EDT normalized to [0..1] (center→boundary), band selection by percentage
  - Outside (>100%): background-only expansion; thickness scaled by each cell’s max internal distance; nearest-cell labeling via background EDT
- ROI-restricted masking:
  - Per-label intersection of Target/Reference mask with Radial mask
  - Cells without any pixels in the Radial mask get empty Target/Reference masks
- Integration and stats:
  - AND of selected masks; per-cell mean/sum for target/reference on AND mask and on full cell region
  - Ratio image limited to AND region, robustly normalized by 1–99 percentiles for visualization; NaN where reference==0



## Outputs 📦

- Masks: `.npy` files for segmentation labels, target mask, reference mask, radial bool mask, radial label mask, AND mask
- CSV: per-cell summary table
- Ratio: Target/Reference as `.npy` (raw, with NaNs where ref==0) and an 8-bit preview image

## Known limitations / next steps ⚠️

- Radial outer >100% includes background near the cell. As of now, T/R uses only pixels with Reference > 0, to avoid NaN/inf. If you want Target/Reference masks computed strictly within ≤100% (inside-cell only), consider adding a toggle to clip ROI at 100%.
- No persistent project/session saving; use the exported `.npy`/`.csv` files to reproduce results.
- The Single image flow computes single-channel statistics only (no ratio). Ratios remain in the Dual images flow.
