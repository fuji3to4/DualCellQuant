# Radial Profile Pixel Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new radial profile analysis function that works in signed pixel units while leaving the existing percent-based analysis and UI behavior unchanged.

**Architecture:** `dualcellquant/radial.py` will gain one shared internal helper that does the preprocessing, per-cell windowing, accumulation, CSV export, and plotting. The existing percent-based function and the new pixel-based function will be thin wrappers around that helper, so the public API stays stable and the pixel version is isolated for personal analysis use.

**Tech Stack:** Python, NumPy, Pandas, SciPy `ndimage`, Matplotlib, PIL

---

### Task 1: Extract shared radial-profile windowing logic

**Files:**
- Modify: `dualcellquant/radial.py:111-700`

- [ ] **Step 1: Refactor the repeated preprocessing and window-accumulation code into one private helper**

```python
def _radial_profile_analysis_impl(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
    start_value: float,
    end_value: float,
    window_size_value: float,
    window_step_value: float,
    pp_bg_enable: bool,
    pp_bg_radius: int,
    pp_norm_enable: bool,
    pp_norm_method: str,
    *,
    bg_mode: str = "rolling",
    bg_dark_pct: float = 5.0,
    manual_tar_bg: float | None = None,
    manual_ref_bg: float | None = None,
    window_bins: int = 1,
    show_errorbars: bool = True,
    ratio_ref_epsilon: float = 0.0,
    axis_label: str,
    axis_unit_label: str,
    inside_transform,
    outside_transform,
):
    ...
```

The helper should keep the current CSV schema for the percent version, but parameterize the axis values and axis label so the percent wrapper and pixel wrapper can both reuse it. Keep the existing ratio, mean, std, sem, and plot behavior intact.

- [ ] **Step 2: Route `radial_profile_analysis()` through the helper without changing its current output columns**

```python
def radial_profile_analysis(...):
    return _radial_profile_analysis_impl(
        ...,
        axis_label="Radial %",
        axis_unit_label="%",
        inside_transform=lambda di, dmax, dist_bg=None: 1.0 - (di / dmax),
        outside_transform=lambda dist_bg, dmax: 1.0 + (dist_bg / dmax),
    )
```

- [ ] **Step 3: Keep the existing percent-based code path producing the same plot labels and CSV columns**

```python
ax1.set_xlabel("Radial % (0=center, 100=boundary)")
```

Run: `python -c "from dualcellquant.radial import radial_profile_analysis"`
Expected: import succeeds after the refactor.

### Task 2: Add a signed-pixel radial profile function

**Files:**
- Modify: `dualcellquant/radial.py:111-700`
- Modify: `dualcellquant/__init__.py:33-88`

- [ ] **Step 1: Add `radial_profile_analysis_px()` as a new public wrapper**

```python
def radial_profile_analysis_px(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
    start_px: float,
    end_px: float,
    window_size_px: float,
    window_step_px: float,
    pp_bg_enable: bool,
    pp_bg_radius: int,
    pp_norm_enable: bool,
    pp_norm_method: str,
    *,
    bg_mode: str = "rolling",
    bg_dark_pct: float = 5.0,
    manual_tar_bg: float | None = None,
    manual_ref_bg: float | None = None,
    window_bins: int = 1,
    show_errorbars: bool = True,
    ratio_ref_epsilon: float = 0.0,
):
    ...
```

Use a signed pixel axis where the cell boundary is `0`, inside is negative, and outside is positive. Keep the returned table shape and plot output consistent with the percent function, but rename the axis columns to `band_start_px`, `band_end_px`, and `center_px`.

- [ ] **Step 2: Encode the signed-pixel transform explicitly inside the wrapper**

```python
inside_transform = lambda di, dmax, dist_bg=None: -(dmax - di)
outside_transform = lambda dist_bg, dmax: dist_bg
```

The wrapper should keep all aggregation logic unchanged; only the axis values and labels should change.

- [ ] **Step 3: Export the new function from `dualcellquant/__init__.py`**

```python
from .radial import (
    radial_mask,
    radial_profile_analysis,
    radial_profile_analysis_px,
    radial_profile_single,
    radial_profile_all_cells,
    compute_radial_peak_difference,
)
```

Run: `python -c "from dualcellquant import radial_profile_analysis_px"`
Expected: import succeeds.

### Task 3: Smoke-test the new pixel function and preserve the old one

**Files:**
- No repo file changes; use a one-off verification command

- [ ] **Step 1: Run a small import-and-shape smoke test against both wrappers**

```powershell
@'
from PIL import Image
import numpy as np
from dualcellquant.radial import radial_profile_analysis, radial_profile_analysis_px

img = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
masks = np.zeros((8, 8), dtype=np.int32)
masks[2:6, 2:6] = 1

res_pct = radial_profile_analysis(img, img, masks, 0, 0, 0, 100, 20, 20, False, 0, False, "none")
res_px = radial_profile_analysis_px(img, img, masks, 0, 0, -4, 4, 2, 2, False, 0, False, "none")

assert res_pct[0].shape[0] > 0
assert res_px[0].shape[0] > 0
assert "center_px" in res_px[0].columns
assert "center_pct" in res_pct[0].columns
print("smoke ok")
'@ | python -
```

Expected: `smoke ok`

