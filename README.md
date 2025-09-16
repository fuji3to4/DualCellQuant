# DualCellQuant

A stepwise Gradio app for per-cell quantification across two fluorescence images with Cellpose-SAM segmentation and flexible masking. This README documents only `dualCellQuant.py`.

## Features

- Cellpose-SAM segmentation (lazy model load, optional GPU)
- Independent, shape-aware Radial mask (optional)
- Target/Reference masking with multiple strategies
	- none / global percentile / global Otsu / per-cell percentile / per-cell Otsu
	- Optional restriction to the Radial ROI (per label)
- Final integration and per-cell statistics
	- AND(Target, Reference[, Radial])
	- Mean/Sum per mask and per whole cell
	- Target/Reference ratio image and CSV export
- Clear overlays with cell IDs and downloadable NumPy arrays for all masks

## Workflow

1) Run Cellpose-SAM segmentation
	- Choose source image/channel and model thresholds
	- Outputs: label mask (.npy), overlays
2) (Optional) Build Radial mask
	- Shape-aware ring between Inner% (0=center) and Outer% (100=boundary)
	- Outer >100% expands outward into background only; labels are propagated via nearest cell
	- Outputs: radial mask (bool .npy), radial labels (.npy), overlay
3) Apply Target mask
	- Choose channel, saturation limit, masking mode, percentile (if applicable), min object size
	- Optionally restrict to the Radial ROI (per-label)
	- Outputs: target mask (.npy), overlay
4) Apply Reference mask
	- Same options as Target; can also use Radial ROI
	- Outputs: reference mask (.npy), overlay
5) Integrate & Quantify
	- Optionally AND with Radial mask
	- Produces per-cell table (CSV), AND mask (.npy), and a Target/Reference ratio image (.npy and preview)

## UI Controls (by section)

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

## Key Implementation Notes

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
	- When enabled, for each cell label L we only evaluate/carry pixels where `radial_labels == L`
- Integration and stats:
	- AND of selected masks; per-cell mean/sum for target/reference on AND mask and on full cell region
	- Ratio image limited to AND region, robustly normalized by 1–99 percentiles for visualization; NaN where reference==0

## Run locally

The app is a single-file Gradio UI. With Poetry:

```pwsh
poetry install
poetry run python .\dualCellQuant.py
```

Or with a plain environment (ensure Python 3.11+ and dependencies in `pyproject.toml`):

```pwsh
python .\dualCellQuant.py
```

Then open the local Gradio URL shown in the terminal.


## Outputs

- Masks: `.npy` files for segmentation labels, target mask, reference mask, radial bool mask, radial label mask, AND mask
- CSV: per-cell summary table
- Ratio: Target/Reference as `.npy` (raw, with NaNs where ref==0) and an 8-bit preview image

## Known limitations / next steps

- Radial outer >100% includes background near the cell. If you want Target/Reference masks computed only within ≤100%, add a toggle to clip ROI at 100%.
- No persistent project/session saving; use the exported `.npy`/`.csv` files to reproduce results.
