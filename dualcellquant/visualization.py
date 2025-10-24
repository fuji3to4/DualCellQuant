"""
Visualization utilities for DualCellQuant.

Includes:
- Overlay generation
- Label annotation
- Plot generation
"""

# Stepwise Cellpose-SAM Gradio App
"""
Stepwise pipeline for Cellpose-SAM with Gradio.

Stages:
1. Run Cellpose segmentation.
2. Build optional radial ROI.
3. Apply Target/Reference masks.
4. Integrate & Quantify (preprocess applied only here).

This allows parameter tuning at each stage before integration.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import tempfile
import math
from skimage import filters, morphology, measure, color, restoration
import scipy.ndimage as ndi
from cellpose import models
import io
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import dualcellquant as dcq

# -----------------------
# Display/label settings
# Note: Use the package-level setting (dcq.LABEL_SCALE) so UI updates apply globally
# -----------------------
# Keep a fallback default only; actual value is read dynamically from the package.
_LABEL_SCALE_FALLBACK: float = 1.8



def _load_font(point_size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/ARIALUNI.TTF",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, point_size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", point_size)
    except Exception:
        return ImageFont.load_default()

def colorize_overlay(image_gray: np.ndarray, masks: np.ndarray, vis_mask: Optional[np.ndarray]) -> Image.Image:
    edges = np.zeros_like(masks, dtype=bool)
    if masks.max() > 0:
        for lab in np.unique(masks):
            if lab == 0:
                continue
            obj = masks == lab
            er = ndi.binary_erosion(obj, iterations=1, border_value=0)
            edges |= obj ^ er
    base = (np.clip(image_gray, 0, 1) * 255).astype(np.uint8)
    overlay = np.stack([base, base, base], axis=2)
    overlay[edges] = [255, 0, 0]
    if vis_mask is not None:
        er = ndi.binary_erosion(vis_mask, iterations=1)
        vis_edges = vis_mask ^ er
        overlay[vis_edges] = [0, 255, 0]
    return Image.fromarray(overlay)

def vivid_label_image(masks: np.ndarray) -> Image.Image:
    lbl_rgb = color.label2rgb(masks, bg_label=0, bg_color=(0, 0, 0), alpha=1.0)
    img = (np.clip(lbl_rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)

def annotate_ids(img: Image.Image, masks: np.ndarray) -> Image.Image:
    try:
        ls = float(getattr(dcq, "LABEL_SCALE", _LABEL_SCALE_FALLBACK))
        if ls <= 0.0:
            return img
    except Exception:
        pass
    draw = ImageDraw.Draw(img)
    w, h = img.size
    base = max(14, int(min(w, h) * 0.035))
    try:
        ls = float(getattr(dcq, "LABEL_SCALE", _LABEL_SCALE_FALLBACK))
    except Exception:
        ls = _LABEL_SCALE_FALLBACK
    fsize = max(12, int(base * ls))
    font = _load_font(fsize)
    props = measure.regionprops(masks)
    for p in props:
        lab = p.label
        cy, cx = p.centroid
        x, y = int(cx), int(cy)
        text = str(lab)
        outline_color = (0, 0, 0)
        off = max(2, fsize // 12)
        for dx, dy in [(-off,0),(off,0),(0,-off),(0,off),(-off,-off),(off,off),(-off,off),(off,-off)]:
            draw.text((x+dx, y+dy), text, fill=outline_color, font=font, anchor="mm")
        draw.text((x, y), text, fill=(255, 255, 0), font=font, anchor="mm")
    return img

def arr01_to_pil_for_preview(arr: np.ndarray) -> Image.Image:
    a = arr
    eps = 1e-6
    try:
        amin = float(np.nanmin(a)); amax = float(np.nanmax(a))
    except Exception:
        amin = 0.0; amax = 0.0
    if amin >= 0.0 and amax <= 1.0:
        clipped = np.clip(a, 0.0, 1.0)
        if clipped.ndim == 2:
            return Image.fromarray((clipped * 255).astype(np.uint8))
        if clipped.ndim == 3 and clipped.shape[2] >= 3:
            out = (np.clip(clipped[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
            return Image.fromarray(out)
    if a.ndim == 2:
        vmin = float(np.nanpercentile(a, 1.0)); vmax = float(np.nanpercentile(a, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
            vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
        denom = (vmax - vmin) if (vmax - vmin) > eps else 1.0
        a_scaled = np.clip((a - vmin) / denom, 0.0, 1.0)
        return Image.fromarray((a_scaled * 255).astype(np.uint8))
    if a.ndim == 3 and a.shape[2] >= 3:
        chs = []
        for i in range(3):
            ch = a[:, :, i]
            vmin = float(np.nanpercentile(ch, 1.0)); vmax = float(np.nanpercentile(ch, 99.0))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= eps:
                vmin, vmax = float(np.nanmin(ch)), float(np.nanmax(ch))
            denom = (vmax - vmin) if (vmax - vmin) > eps else 1.0
            chs.append(np.clip((ch - vmin) / denom, 0.0, 1.0))
        out = (np.stack(chs, axis=2) * 255).astype(np.uint8)
        return Image.fromarray(out)
    a_clipped = np.clip(a, 0.0, 1.0)
    return Image.fromarray((a_clipped * 255).astype(np.uint8))

def save_bool_mask_tiff(mask: np.ndarray, stem: str) -> str:
    arr = (mask.astype(np.uint8) * 255)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(arr).save(tmp.name)
    return tmp.name

def save_label_tiff(labels: np.ndarray, stem: str) -> str:
    maxv = int(labels.max()) if labels.size > 0 else 0
    if maxv <= 65535:
        out = labels.astype(np.uint16, copy=False)
    else:
        out = labels.astype(np.float32, copy=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{stem}.tif")
    Image.fromarray(out).save(tmp.name)
    return tmp.name

def plot_radial_profile_with_peaks(
    df: pd.DataFrame,
    peak_df: pd.DataFrame = None,
    label_filter: int | str = "All",
    window_bins: int = 1,
    show_errorbars: bool = True,
    show_ratio: bool = True,
    title_suffix: str = "",
) -> Image.Image:
    """
    Generate radial profile plot with optional peak difference markers.
    
    Args:
        df: Radial profile DataFrame with columns: label, center_pct, mean_target, mean_reference, etc.
        peak_df: Peak difference DataFrame with columns: label, max_target_center_pct, max_reference_center_pct, difference_pct, etc.
        label_filter: "All" for pooled average, or specific label number for single-cell plot
        window_bins: Smoothing window size (1 = no smoothing)
        show_errorbars: Whether to show error bars (SEM)
        show_ratio: Whether to show T/R ratio on secondary axis
        title_suffix: Optional text to append to plot title
    
    Returns:
        PIL Image of the plot
    """
    if df is None or df.empty:
        # Return blank plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
    
    # Determine if plotting All or single label
    is_all = str(label_filter) == "All"
    
    if is_all:
        # Pool data across all labels, weighted by pixel count
        bins = np.sort(df["center_pct"].dropna().unique())
        M_t = []; SEM_t = []; M_r = []; SEM_r = []; M_ratio = []; SEM_ratio = []
        for c in bins:
            g = df[df["center_pct"] == c]
            # Target pooled
            gi = g.dropna(subset=["mean_target", "std_target", "count_px"]) if not g.empty else g
            n = gi["count_px"].to_numpy(dtype=float)
            m = gi["mean_target"].to_numpy(dtype=float)
            s = gi["std_target"].to_numpy(dtype=float)
            N = float(np.nansum(n)) if gi is not None else 0.0
            if N > 0:
                M = float(np.nansum(n * m) / N)
                if N > 1 and gi.shape[0] > 0:
                    SS_within = np.nansum((n - 1) * (s ** 2))
                    SS_between = np.nansum(n * ((m - M) ** 2))
                    var = (SS_within + SS_between) / (N - 1)
                    sem = np.sqrt(var) / np.sqrt(N)
                else:
                    sem = np.nan
            else:
                M = np.nan; sem = np.nan
            M_t.append(M); SEM_t.append(sem)
            # Reference pooled
            gi = g.dropna(subset=["mean_reference", "std_reference", "count_px"]) if not g.empty else g
            n = gi["count_px"].to_numpy(dtype=float)
            m = gi["mean_reference"].to_numpy(dtype=float)
            s = gi["std_reference"].to_numpy(dtype=float)
            N = float(np.nansum(n)) if gi is not None else 0.0
            if N > 0:
                M = float(np.nansum(n * m) / N)
                if N > 1 and gi.shape[0] > 0:
                    SS_within = np.nansum((n - 1) * (s ** 2))
                    SS_between = np.nansum(n * ((m - M) ** 2))
                    var = (SS_within + SS_between) / (N - 1)
                    sem = np.sqrt(var) / np.sqrt(N)
                else:
                    sem = np.nan
            else:
                M = np.nan; sem = np.nan
            M_r.append(M); SEM_r.append(sem)
            # Ratio pooled
            if "count_ratio_px" in g.columns:
                gi = g.dropna(subset=["mean_ratio_T_over_R"]) if not g.empty else g
                nr = gi.get("count_ratio_px", pd.Series(np.zeros(len(gi)), index=gi.index)).to_numpy(dtype=float)
                mr = gi["mean_ratio_T_over_R"].to_numpy(dtype=float)
                sr = gi.get("std_ratio_T_over_R", pd.Series(np.nan, index=gi.index)).to_numpy(dtype=float)
                NR = float(np.nansum(nr)) if gi is not None else 0.0
                if NR > 0:
                    MR = float(np.nansum(nr * mr) / NR)
                    if NR > 1 and gi.shape[0] > 0:
                        SS_within_r = np.nansum((nr - 1) * (sr ** 2))
                        SS_between_r = np.nansum(nr * ((mr - MR) ** 2))
                        var_r = (SS_within_r + SS_between_r) / (NR - 1)
                        sem_r = np.sqrt(var_r) / np.sqrt(NR)
                    else:
                        sem_r = np.nan
                else:
                    MR = np.nan; sem_r = np.nan
            else:
                MR = np.nan; sem_r = np.nan
            M_ratio.append(MR); SEM_ratio.append(sem_r)
        
        x = bins
        ma_t = _moving_average_nan(np.array(M_t, dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else np.array(M_t, dtype=float)
        ma_r = _moving_average_nan(np.array(M_r, dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else np.array(M_r, dtype=float)
        ma_ratio = _moving_average_nan(np.array(M_ratio, dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else np.array(M_ratio, dtype=float)
        sem_t_arr = np.array(SEM_t, dtype=float)
        sem_r_arr = np.array(SEM_r, dtype=float)
        sem_ratio_arr = np.array(SEM_ratio, dtype=float)
        plot_label_text = "All cells"
    else:
        # Single label
        try:
            lab_int = int(label_filter)
        except:
            lab_int = None
        if lab_int is None:
            # Return blank
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Invalid label", ha="center", va="center", transform=ax.transAxes)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).copy()
            buf.close()
            return img
        df_lab = df[df["label"] == lab_int].copy()
        if df_lab.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"No data for label {lab_int}", ha="center", va="center", transform=ax.transAxes)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).copy()
            buf.close()
            return img
        x = df_lab["center_pct"].to_numpy(dtype=float)
        ma_t = _moving_average_nan(df_lab["mean_target"].to_numpy(dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else df_lab["mean_target"].to_numpy(dtype=float)
        ma_r = _moving_average_nan(df_lab["mean_reference"].to_numpy(dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else df_lab["mean_reference"].to_numpy(dtype=float)
        ma_ratio = _moving_average_nan(df_lab["mean_ratio_T_over_R"].to_numpy(dtype=float), int(window_bins)) if window_bins and int(window_bins) > 1 else df_lab["mean_ratio_T_over_R"].to_numpy(dtype=float)
        sem_t_arr = df_lab.get("sem_target", df_lab.get("std_target", pd.Series(np.nan, index=df_lab.index))).to_numpy(dtype=float)
        sem_r_arr = df_lab.get("sem_reference", df_lab.get("std_reference", pd.Series(np.nan, index=df_lab.index))).to_numpy(dtype=float)
        sem_ratio_arr = df_lab.get("sem_ratio_T_over_R", df_lab.get("std_ratio_T_over_R", pd.Series(np.nan, index=df_lab.index))).to_numpy(dtype=float)
        plot_label_text = f"Label {lab_int}"
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(7, 5))
    if show_errorbars:
        ax1.errorbar(x, ma_t, yerr=sem_t_arr, fmt='-o', ms=3, capsize=2, label="Target", color="tab:red", alpha=0.9)
        ax1.errorbar(x, ma_r, yerr=sem_r_arr, fmt='-o', ms=3, capsize=2, label="Reference", color="tab:blue", alpha=0.9)
    else:
        ax1.plot(x, ma_t, label="Target", color="tab:red")
        ax1.plot(x, ma_r, label="Reference", color="tab:blue")
    ax1.set_xlabel("Radial % (0=center, 100=boundary)")
    ax1.set_ylabel("Mean intensity")
    ax1.grid(True, alpha=0.3)
    
    # Add peak markers if peak_df is provided
    peak_annotation_text = None
    if peak_df is not None and not peak_df.empty:
        if is_all:
            # For "All", we can't show individual peaks since they differ per cell
            # But we could show a summary or skip
            pass
        else:
            # For single label, show the peak positions
            peak_row = peak_df[peak_df["label"] == lab_int]
            if not peak_row.empty:
                peak_row = peak_row.iloc[0]
                max_t_pct = peak_row.get("max_target_center_pct", np.nan)
                max_r_pct = peak_row.get("max_reference_center_pct", np.nan)
                diff_pct = peak_row.get("difference_pct", np.nan)
                diff_px = peak_row.get("difference_px", np.nan)
                diff_um = peak_row.get("difference_um", np.nan)
                
                if np.isfinite(max_t_pct):
                    ax1.axvline(max_t_pct, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Target peak: {max_t_pct:.1f}%")
                if np.isfinite(max_r_pct):
                    ax1.axvline(max_r_pct, color="blue", linestyle="--", alpha=0.5, linewidth=1.5, label=f"Reference peak: {max_r_pct:.1f}%")
                
                # Build annotation text for difference
                if np.isfinite(diff_pct):
                    text_lines = [f"Δ = {diff_pct:.1f}%"]
                    if np.isfinite(diff_px):
                        text_lines.append(f"  (≈{diff_px:.2f} px)")
                    if np.isfinite(diff_um):
                        text_lines.append(f"  (≈{diff_um:.2f} μm)")
                    peak_annotation_text = "\n".join(text_lines)
    
    # Plot ratio on secondary axis if requested
    if show_ratio:
        ax2 = ax1.twinx()
        if show_errorbars:
            ax2.errorbar(x, ma_ratio, yerr=sem_ratio_arr, fmt='-s', ms=3, capsize=2, label="T/R", color="tab:green", alpha=0.9)
        else:
            ax2.plot(x, ma_ratio, label="T/R", color="tab:green", linestyle="--")
        ax2.set_ylabel("Mean ratio (T/R)")
        
        # Combined legend - position at upper left to avoid peak markers
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        # Legend for ax1 only
        ax1.legend(loc="upper left", fontsize=8)
    
    # Add peak difference annotation after legend (so it appears on top)
    if peak_annotation_text is not None:
        ax1.text(0.98, 0.95, peak_annotation_text, transform=ax1.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                fontsize=9, family="monospace")
    
    plot_title = f"Radial Profile - {plot_label_text}"
    if title_suffix:
        plot_title += f" {title_suffix}"
    ax1.set_title(plot_title, fontsize=10)
    
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

def _moving_average_nan(y: np.ndarray, window_bins: int) -> np.ndarray:
    """NaNを無視した移動平均。window_binsは1以上の奇数を推奨（偶数は繰上げて奇数化）。"""
    if y is None:
        return None
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    w = int(max(1, window_bins))
    if w % 2 == 0:
        w += 1
    if w == 1:
        return y.copy()
    half = w // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        win = y[a:b]
        finite = np.isfinite(win)
        if np.any(finite):
            out[i] = float(np.mean(win[finite]))
    return out


def save_radial_profile_grid_png(
    df: pd.DataFrame,
    peak_df: pd.DataFrame | None = None,
    *,
    window_bins: int = 1,
    show_errorbars: bool = True,
    show_ratio: bool = True,
    labels: list[int] | None = None,
    cols: int = 3,
    tile_width: int = 700,
    title_suffix: str = "",
) -> str:
    """
    Create a grid image (3 columns by variable rows) of per-cell radial profiles and save as PNG.

    Args:
        df: Radial profile DataFrame (contains column 'label').
        peak_df: Optional peak-difference DataFrame to overlay peak markers.
        window_bins: Smoothing bins passed to plot function.
        show_errorbars: Whether to draw SEM bars.
        labels: Specific label list to plot. If None, uses all labels in df (sorted).
        cols: Number of columns in the grid (default 3).
        tile_width: Resize width for each tile plot (keeps aspect ratio).
        title_suffix: Optional suffix appended to each tile plot title.

    Returns:
        str: Temporary PNG file path containing the grid image.
    """
    if df is None or df.empty:
        # Create a small placeholder image
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_grid.png")
        img.save(tmp.name)
        return tmp.name

    # Resolve labels to plot
    if labels is None:
        try:
            labels = sorted(int(x) for x in pd.Series(df["label"].unique()).dropna().to_list())
        except Exception:
            labels = []

    if len(labels) == 0:
        # Same placeholder as above
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No labels found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_grid.png")
        img.save(tmp.name)
        return tmp.name

    # Build per-label images
    tiles: list[Image.Image] = []
    for lab in labels:
        try:
            im = plot_radial_profile_with_peaks(
                df, peak_df, label_filter=int(lab), window_bins=int(window_bins),
                show_errorbars=bool(show_errorbars), show_ratio=bool(show_ratio), title_suffix=title_suffix,
            )
        except Exception:
            # Fallback blank tile on error
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"Plot failed: {lab}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).copy()
            buf.close()
        # Resize tile to target width
        if tile_width and im.width != tile_width:
            scale = tile_width / float(im.width)
            new_h = max(1, int(round(im.height * scale)))
            im = im.resize((int(tile_width), int(new_h)), resample=Image.BICUBIC)
        tiles.append(im)

    if cols <= 0:
        cols = 3
    rows = int(math.ceil(len(tiles) / float(cols)))
    tile_w = max((im.width for im in tiles), default=tile_width or 700)
    tile_h = max((im.height for im in tiles), default=int(tile_width * 0.66) if tile_width else 500)

    grid_w = cols * tile_w
    grid_h = rows * tile_h
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = r * tile_h
        # Center the tile in its cell
        off_x = x + max(0, (tile_w - tile.width) // 2)
        off_y = y + max(0, (tile_h - tile.height) // 2)
        canvas.paste(tile, (off_x, off_y))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_grid.png")
    canvas.save(tmp.name)
    return tmp.name


def build_radial_profile_grid_image(
    df: pd.DataFrame,
    peak_df: pd.DataFrame | None = None,
    *,
    window_bins: int = 1,
    show_errorbars: bool = True,
    show_ratio: bool = True,
    labels: list[int] | None = None,
    cols: int = 3,
    tile_width: int = 700,
    title_suffix: str = "",
) -> Image.Image:
    """
    Create a grid PIL Image of per-cell radial profiles (3 columns, variable rows).
    """
    # Reuse the PNG builder then load as PIL to avoid code duplication, but keep in-memory result.
    path = save_radial_profile_grid_png(
        df, peak_df,
        window_bins=window_bins,
        show_errorbars=show_errorbars,
        show_ratio=show_ratio,
        labels=labels,
        cols=cols,
        tile_width=tile_width,
        title_suffix=title_suffix,
    )
    try:
        img = Image.open(path).copy()
        return img
    except Exception:
        # Fallback blank
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Failed to build grid image", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

