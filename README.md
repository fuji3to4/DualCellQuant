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
- uv

```pwsh
uv sync
uv run python dualCellQuant.py
```

- Pip

```pwsh
pip install .
python dualCellQuant.py
```

- Pip (direct from GitHub)

```pwsh
pip install "git+https://github.com/fuji3to4/DualCellQuant.git"
```

- Optional: mount under FastAPI at `/dualcellquant`: 

```pwsh
uv run uvicorn serve:app
```

[http://127.0.0.1:8000/dualcellquant](http://127.0.0.1:8000/dualcellquant)

### Troubleshooting: CUDA issues

If CUDA is not detected or PyTorch is not using GPU:

1. Check CUDA version: `nvidia-smi`
2. Reinstall torch/torchvision matching your CUDA version (replace `cu130` with your version, e.g., `cu126`, `cu128`):

```pwsh
uv pip uninstall -y torch torchvision
uv pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
```

3. Verify installation:

```pwsh
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
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



## Citation

If you use DualCellQuant in your work, please cite both the software and the paper:

### Paper

Fujii, S., Takaki, K., & Sueda, S. (2026). Dual-color image analysis for quantifying fluorescence intensity in plasma membrane region of cells. *Analytical Sciences*. https://doi.org/10.1007/s44211-026-00908-y

👉 **[examples/run_reproduction_pipeline.ipynb](examples/run_reproduction_pipeline.ipynb)** - Complete batch processing pipeline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuji3to4/dualCellQuant/blob/main/examples/run_reproduction_pipeline.ipynb)


### Software

Fujii, S. (2026). *fuji3to4/DualCellQuant: v1.0.0* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18347379

BibTeX:

```bibtex
@article{fujii_2026_dual_color_image_analysis,
  author       = {Fujii, Satoshi and Takaki, Keita and Sueda, Sinji},
  title        = {Dual-color image analysis for quantifying fluorescence intensity in plasma membrane region of cells},
  journal      = {Analytical Sciences},
  year         = {2026},
  doi          = {10.1007/s44211-026-00908-y},
  url          = {https://doi.org/10.1007/s44211-026-00908-y}
}

@misc{fujii_2026_dualcellquant_notebook,
  author       = {Fujii, Satoshi},
  title        = {DualCellQuant - Batch Processing Workflow (Paper Reproduction)},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/fuji3to4/dualCellQuant/blob/main/examples/run_reproduction_pipeline.ipynb}}
}

@software{fujii_2026_dualcellquant_v1_0_0,
  author       = {Fujii, Satoshi},
  title        = {fuji3to4/DualCellQuant: v1.0.0},
  year         = {2026},
  version      = {v1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18347379},
  url          = {https://doi.org/10.5281/zenodo.18347379}
}
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
