"""
Radial analysis functionality for DualCellQuant.

Includes:
- Radial mask generation
- Radial profile analysis
- Peak difference computation
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
from skimage import filters, morphology, measure, color, restoration
import scipy.ndimage as ndi
from cellpose import models
import io
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -----------------------
# Display/label settings (adjustable via UI)
# -----------------------
LABEL_SCALE: float = 1.8



from .core import (
    extract_single_channel, pil_to_numpy, pil_to_numpy_native,
    preprocess_for_processing, cleanup_mask,
)
from .visualization import (
    arr01_to_pil_for_preview, _moving_average_nan,
    colorize_overlay, annotate_ids, save_bool_mask_tiff, save_label_tiff,
)

def radial_mask(
    masks: np.ndarray,
    inner_pct: float,
    outer_pct: float,
    min_obj_size: int,
):
    if masks is None:
        raise ValueError("Segmentation masks not provided. Run segmentation first.")
    labels = np.unique(masks); labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError("No cells found in masks.")
    H, W = masks.shape
    bg = masks == 0
    props_map = {p.label: p for p in measure.regionprops(masks)}
    radial_total = np.zeros_like(masks, dtype=bool)
    radial_labels = np.zeros_like(masks, dtype=masks.dtype)
    dmax_table = {}
    for lab in labels:
        cell = masks == lab
        p = props_map.get(int(lab))
        if p is None or p.area <= 0:
            continue
        di = ndi.distance_transform_edt(cell)
        dmax = float(di.max())
        if dmax <= 0:
            radial_total |= cell
            continue
        dmax_table[int(lab)] = dmax
        t = 1.0 - (di / dmax)
        rin_t = max(0.0, float(inner_pct)) / 100.0
        rout_t = max(0.0, float(outer_pct)) / 100.0
        if rin_t > rout_t:
            rin_t, rout_t = rout_t, rin_t
        inside_band = (t >= rin_t) & (t <= min(rout_t, 1.0)) & cell
        radial_cell = cleanup_mask(inside_band.copy(), int(min_obj_size))
        radial_total |= radial_cell
        radial_labels[radial_cell] = int(lab)
    extra_frac = max(0.0, float(outer_pct) / 100.0 - 1.0)
    if extra_frac > 0.0 and len(dmax_table) > 0:
        fg = masks > 0
        dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
        nearest_label = masks[idx[0], idx[1]]
        max_lab = int(masks.max())
        extra_table = np.zeros(max_lab + 1, dtype=np.float32)
        for lab, dmax in dmax_table.items():
            extra_table[int(lab)] = extra_frac * float(dmax)
        outside_bool = (~fg) & (nearest_label > 0) & (dist_bg > 0) & (dist_bg <= extra_table[nearest_label])
        add_mask = outside_bool & bg
        radial_total |= add_mask
        to_set = (radial_labels == 0) & add_mask
        radial_labels[to_set] = nearest_label[to_set].astype(radial_labels.dtype)
    overlay = colorize_overlay((masks > 0).astype(np.float32), masks, radial_total)
    overlay = annotate_ids(overlay, masks)
    rad_bool_tiff = save_bool_mask_tiff(radial_total, "radial_mask")
    rad_lbl_tiff = save_label_tiff(radial_labels, "radial_labels")
    return overlay, radial_total, radial_labels, rad_bool_tiff, rad_lbl_tiff

def radial_profile_analysis(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
    start_pct: float,
    end_pct: float,
    window_size_pct: float,
    window_step_pct: float,
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
    if masks is None:
        raise ValueError("Segmentation masks not provided. Run segmentation first.")
    labels = np.unique(masks); labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError("No cells found in masks.")
    # Preprocess images for measurement in native scale
    tgt_nat = preprocess_for_processing(
        target_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_tar_bg,
    )
    ref_nat = preprocess_for_processing(
        reference_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_ref_bg,
    )
    tgt_gray = extract_single_channel(tgt_nat, tgt_chan)
    ref_gray = extract_single_channel(ref_nat, ref_chan)

    # Clamp window settings
    s = float(start_pct); e = float(end_pct)
    wsize = float(window_size_pct); wstep = float(window_step_pct)
    if wsize <= 0: wsize = 10.0
    if wstep <= 0: wstep = 5.0
    if e < s: s, e = e, s
    
    # Generate window starts from s to e-wsize (inclusive) with step wstep
    window_starts = []
    current = s
    while current <= e - wsize + 1e-6:
        window_starts.append(current)
        current += wstep
    
    if len(window_starts) == 0:
        raise ValueError(f"Invalid window settings: no windows formed (start={s}, end={e}, size={wsize}, step={wstep})")
    
    windows = [(start, start + wsize) for start in window_starts]
    nwindows = len(windows)

    # Precompute outside distance and nearest labels for extension >100%
    fg = masks > 0
    dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
    nearest_label = masks[idx[0], idx[1]]

    # Accumulators per window
    sum_t = np.zeros(nwindows, dtype=np.float64)
    sum_r = np.zeros(nwindows, dtype=np.float64)
    cnt = np.zeros(nwindows, dtype=np.int64)
    # second moment for std
    sum_t2 = np.zeros(nwindows, dtype=np.float64)
    sum_r2 = np.zeros(nwindows, dtype=np.float64)

    # Loop per cell, compute inside/outside normalized radii and accumulate
    for lab in labels:
        cell = (masks == lab)
        if not np.any(cell):
            continue
        di = ndi.distance_transform_edt(cell)
        dmax = float(di.max())
        if dmax <= 0:
            continue
        # inside: t in [0,1]
        tin = 1.0 - (di / dmax)
        # outside assigned to this label
        out_lab = (~fg) & (nearest_label == lab)
        if np.any(out_lab):
            tout = np.zeros_like(tin, dtype=np.float32)
            tout[out_lab] = 1.0 + (dist_bg[out_lab].astype(np.float32) / dmax)
        else:
            tout = None

        # Accumulate per window
        for k in range(nwindows):
            a_pct, b_pct = windows[k]
            a = a_pct / 100.0
            b = b_pct / 100.0
            # include both endpoints for the window
            idx_in = cell & (tin >= a) & (tin <= b)
            if tout is not None:
                idx_out = out_lab & (tout >= a) & (tout <= b)
            else:
                idx_out = None
            if idx_out is not None:
                idx_band = idx_in | idx_out
            else:
                idx_band = idx_in
            n = int(np.count_nonzero(idx_band))
            if n == 0:
                continue
            cnt[k] += n
            vals_t = tgt_gray[idx_band]
            vals_r = ref_gray[idx_band]
            sum_t[k] += float(np.sum(vals_t))
            sum_r[k] += float(np.sum(vals_r))
            sum_t2[k] += float(np.sum(vals_t.astype(np.float64) ** 2))
            sum_r2[k] += float(np.sum(vals_r.astype(np.float64) ** 2))

    # Build results
    center_pct = np.array([(a + b) / 2.0 for a, b in windows])
    mean_t = np.where(cnt > 0, sum_t / cnt, np.nan)
    mean_r = np.where(cnt > 0, sum_r / cnt, np.nan)
    # unbiased variance estimate; guard small n
    var_t = np.where(cnt > 1, (sum_t2 - (sum_t ** 2) / cnt) / (cnt - 1), np.nan)
    var_r = np.where(cnt > 1, (sum_r2 - (sum_r ** 2) / cnt) / (cnt - 1), np.nan)
    std_t = np.sqrt(var_t)
    std_r = np.sqrt(var_r)
    sem_t = np.where(cnt > 0, std_t / np.sqrt(cnt), np.nan)
    sem_r = np.where(cnt > 0, std_r / np.sqrt(cnt), np.nan)
    # Ratio on pixels where reference > 0: approximate using means (optionally) or recompute per-pixel
    # Here we use mean of per-pixel ratio by sampling mask again per window for better fidelity
    mean_ratio = np.full(nwindows, np.nan, dtype=float)
    std_ratio = np.full(nwindows, np.nan, dtype=float)
    cnt_ratio = np.zeros(nwindows, dtype=np.int64)
    eps_ratio = max(1e-12, float(ratio_ref_epsilon))
    for k in range(nwindows):
        a_pct, b_pct = windows[k]
        a = a_pct / 100.0
        b = b_pct / 100.0
        acc = []
        for lab in labels:
            cell = (masks == lab)
            if not np.any(cell):
                continue
            di = ndi.distance_transform_edt(cell)
            dmax = float(di.max())
            if dmax <= 0:
                continue
            tin = 1.0 - (di / dmax)
            out_lab = (~fg) & (nearest_label == lab)
            tout = None
            if np.any(out_lab):
                tout = np.zeros_like(tin, dtype=np.float32)
                tout[out_lab] = 1.0 + (dist_bg[out_lab].astype(np.float32) / dmax)
            idx_in = cell & (tin >= a) & (tin <= b)
            if tout is not None:
                idx_out = out_lab & (tout >= a) & (tout <= b)
                idx_band = idx_in | idx_out
            else:
                idx_band = idx_in
            if np.count_nonzero(idx_band) == 0:
                continue
            tv = tgt_gray[idx_band].astype(np.float64)
            rv = ref_gray[idx_band].astype(np.float64)
            rr = (tv + eps_ratio) / (rv + eps_ratio)
            acc.append(rr)
        if len(acc) > 0:
            allv = np.concatenate(acc)
            if allv.size:
                mean_ratio[k] = float(np.mean(allv))
                std_ratio[k] = float(np.std(allv, ddof=1)) if allv.size > 1 else np.nan
                cnt_ratio[k] = int(allv.size)

    # Table
    band_starts = [a for a, b in windows]
    band_ends = [b for a, b in windows]
    df = pd.DataFrame({
        "band_start_pct": band_starts,
        "band_end_pct": band_ends,
        "center_pct": center_pct,
        "count_px": cnt,
        "mean_target": mean_t,
        "mean_reference": mean_r,
        "std_target": std_t,
        "std_reference": std_r,
        "sem_target": sem_t,
        "sem_reference": sem_r,
        "mean_ratio_T_over_R": mean_ratio,
        "std_ratio_T_over_R": std_ratio,
        "sem_ratio_T_over_R": np.where(cnt_ratio > 0, std_ratio / np.sqrt(cnt_ratio), np.nan),
        "count_ratio_px": cnt_ratio,
    })
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile.csv")
    df.to_csv(tmp_csv.name, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ma_t = _moving_average_nan(mean_t, int(window_bins)) if window_bins and window_bins > 1 else mean_t
    ma_r = _moving_average_nan(mean_r, int(window_bins)) if window_bins and window_bins > 1 else mean_r
    # plot with optional error bars (SEM)
    if show_errorbars:
        ax1.errorbar(center_pct, ma_t, yerr=sem_t, fmt='-o', ms=3, capsize=2, label="Target", color="tab:red", alpha=0.9)
        ax1.errorbar(center_pct, ma_r, yerr=sem_r, fmt='-o', ms=3, capsize=2, label="Reference", color="tab:blue", alpha=0.9)
    else:
        ax1.plot(center_pct, ma_t, label="Target", color="tab:red")
        ax1.plot(center_pct, ma_r, label="Reference", color="tab:blue")
    ax1.set_xlabel("Radial % (0=center, 100=boundary)")
    ax1.set_ylabel("Mean intensity")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ma_ratio = _moving_average_nan(mean_ratio, int(window_bins)) if window_bins and window_bins > 1 else mean_ratio
    if show_errorbars:
        ax2.errorbar(center_pct, ma_ratio, yerr=np.where(cnt_ratio > 0, std_ratio / np.sqrt(cnt_ratio), np.nan), fmt='-s', ms=3, capsize=2, label="T/R", color="tab:green", alpha=0.9)
    else:
        ax2.plot(center_pct, ma_ratio, label="T/R", color="tab:green", linestyle="--")
    ax2.set_ylabel("Mean ratio (T/R)")
    # Build combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    prof_plot = Image.open(buf).copy()
    buf.close()
    return df, tmp_csv.name, prof_plot

def radial_profile_single(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    label: int,
    tgt_chan: int,
    ref_chan: int,
    start_pct: float,
    end_pct: float,
    window_size_pct: float,
    window_step_pct: float,
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
    # Preprocess once
    tgt_nat = preprocess_for_processing(
        target_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_tar_bg,
    )
    ref_nat = preprocess_for_processing(
        reference_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_ref_bg,
    )
    tgt_gray = extract_single_channel(tgt_nat, tgt_chan)
    ref_gray = extract_single_channel(ref_nat, ref_chan)
    labels_all = np.unique(masks); labels_all = labels_all[labels_all > 0]
    if int(label) not in set(map(int, labels_all.tolist())):
        raise ValueError(f"Label {label} not found in masks")
    lab = int(label)
    cell = (masks == lab)
    if not np.any(cell):
        raise ValueError(f"Label {lab} has no pixels")
    di = ndi.distance_transform_edt(cell)
    dmax = float(di.max())
    if dmax <= 0:
        raise ValueError("Invalid cell (zero max distance)")
    tin = 1.0 - (di / dmax)
    # outside ring assigned to this label
    fg = masks > 0
    dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
    nearest_label = masks[idx[0], idx[1]]
    out_lab = (~fg) & (nearest_label == lab)
    tout = None
    if np.any(out_lab):
        tout = np.zeros_like(tin, dtype=np.float32)
        tout[out_lab] = 1.0 + (dist_bg[out_lab].astype(np.float32) / dmax)

    # Sliding windows
    s = float(start_pct); e = float(end_pct)
    wsize = float(window_size_pct); wstep = float(window_step_pct)
    if wsize <= 0: wsize = 10.0
    if wstep <= 0: wstep = 5.0
    if e < s: s, e = e, s
    
    # Generate window starts from s to e-wsize (inclusive) with step wstep
    window_starts = []
    current = s
    while current <= e - wsize + 1e-6:
        window_starts.append(current)
        current += wstep
    
    if len(window_starts) == 0:
        raise ValueError(f"Invalid window settings: no windows formed (start={s}, end={e}, size={wsize}, step={wstep})")
    
    windows = [(start, start + wsize) for start in window_starts]
    nwindows = len(windows)

    rows = []
    for k in range(nwindows):
        a_pct, b_pct = windows[k]
        a = a_pct / 100.0
        b = b_pct / 100.0
        # include both endpoints for the window
        idx_in = cell & (tin >= a) & (tin <= b)
        if tout is not None:
            idx_out = out_lab & (tout >= a) & (tout <= b)
            idx_band = idx_in | idx_out
        else:
            idx_band = idx_in
        n = int(np.count_nonzero(idx_band))
        if n == 0:
            rows.append({
                "label": lab,
                "band_start_pct": a_pct,
                "band_end_pct": b_pct,
                "center_pct": (a_pct + b_pct) / 2.0,
                "count_px": 0,
                "mean_target": np.nan,
                "mean_reference": np.nan,
                "std_target": np.nan,
                "std_reference": np.nan,
                "sem_target": np.nan,
                "sem_reference": np.nan,
                "mean_ratio_T_over_R": np.nan,
                "std_ratio_T_over_R": np.nan,
                "sem_ratio_T_over_R": np.nan,
                "count_ratio_px": 0,
            })
            continue
        vals_t = tgt_gray[idx_band].astype(np.float64)
        vals_r = ref_gray[idx_band].astype(np.float64)
        mean_t = float(np.mean(vals_t))
        mean_r = float(np.mean(vals_r))
        std_t = float(np.std(vals_t, ddof=1)) if n > 1 else np.nan
        std_r = float(np.std(vals_r, ddof=1)) if n > 1 else np.nan
        sem_t = float(std_t / np.sqrt(n)) if n > 1 else np.nan
        sem_r = float(std_r / np.sqrt(n)) if n > 1 else np.nan
        r = ref_gray[idx_band].astype(np.float64)
        t = tgt_gray[idx_band].astype(np.float64)
        eps = max(1e-12, float(ratio_ref_epsilon))
        rr = (t + eps) / (r + eps)
        ratio_mean = float(np.mean(rr)) if rr.size > 0 else np.nan
        ratio_std = float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan
        ratio_sem = float(ratio_std / np.sqrt(rr.size)) if rr.size > 1 else (0.0 if rr.size == 1 else np.nan)
        ratio_cnt = int(rr.size)
        rows.append({
            "label": lab,
            "band_start_pct": a_pct,
            "band_end_pct": b_pct,
            "center_pct": (a_pct + b_pct) / 2.0,
            "count_px": n,
            "mean_target": mean_t,
            "mean_reference": mean_r,
            "std_target": std_t,
            "std_reference": std_r,
            "sem_target": sem_t,
            "sem_reference": sem_r,
            "mean_ratio_T_over_R": ratio_mean,
            "std_ratio_T_over_R": ratio_std,
            "sem_ratio_T_over_R": ratio_sem,
            "count_ratio_px": ratio_cnt,
        })

    df = pd.DataFrame(rows)
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=f"_radial_profile_label_{lab}.csv")
    df.to_csv(tmp_csv.name, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ma_t = _moving_average_nan(df["mean_target"].to_numpy(dtype=float), int(window_bins)) if window_bins and window_bins > 1 else df["mean_target"].to_numpy(dtype=float)
    ma_r = _moving_average_nan(df["mean_reference"].to_numpy(dtype=float), int(window_bins)) if window_bins and window_bins > 1 else df["mean_reference"].to_numpy(dtype=float)
    x = df["center_pct"].to_numpy(dtype=float)
    if show_errorbars:
        ax1.errorbar(x, ma_t, yerr=df["sem_target"].to_numpy(dtype=float), fmt='-o', ms=3, capsize=2, label=f"Target (L{lab})", color="tab:red", alpha=0.9)
        ax1.errorbar(x, ma_r, yerr=df["sem_reference"].to_numpy(dtype=float), fmt='-o', ms=3, capsize=2, label=f"Reference (L{lab})", color="tab:blue", alpha=0.9)
    else:
        ax1.plot(x, ma_t, label=f"Target (L{lab})", color="tab:red")
        ax1.plot(x, ma_r, label=f"Reference (L{lab})", color="tab:blue")
    ax1.set_xlabel("Radial % (0=center, 100=boundary)")
    ax1.set_ylabel("Mean intensity")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ma_ratio = _moving_average_nan(df["mean_ratio_T_over_R"].to_numpy(dtype=float), int(window_bins)) if window_bins and window_bins > 1 else df["mean_ratio_T_over_R"].to_numpy(dtype=float)
    if show_errorbars and "sem_ratio_T_over_R" in df.columns:
        ax2.errorbar(x, ma_ratio, yerr=df["sem_ratio_T_over_R"].to_numpy(dtype=float), fmt='-s', ms=3, capsize=2, label="T/R", color="tab:green", alpha=0.9)
    else:
        ax2.plot(x, ma_ratio, label="T/R", color="tab:green", linestyle="--")
    ax2.set_ylabel("Mean ratio (T/R)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    plot_img = Image.open(buf).copy()
    buf.close()
    return df, tmp_csv.name, plot_img

def radial_profile_all_cells(
    target_img: Image.Image,
    reference_img: Image.Image,
    masks: np.ndarray,
    tgt_chan: int,
    ref_chan: int,
    start_pct: float,
    end_pct: float,
    window_size_pct: float,
    window_step_pct: float,
    pp_bg_enable: bool,
    pp_bg_radius: int,
    pp_norm_enable: bool,
    pp_norm_method: str,
    *,
    bg_mode: str = "rolling",
    bg_dark_pct: float = 5.0,
    manual_tar_bg: float | None = None,
    manual_ref_bg: float | None = None,
    ratio_ref_epsilon: float = 0.0,
):
    # Preprocess once
    tgt_nat = preprocess_for_processing(
        target_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_tar_bg,
    )
    ref_nat = preprocess_for_processing(
        reference_img, use_native_scale=True,
        bg_enable=pp_bg_enable, bg_radius=pp_bg_radius, bg_mode=bg_mode, bg_dark_pct=bg_dark_pct,
        norm_enable=pp_norm_enable, norm_method=pp_norm_method,
        manual_background=manual_ref_bg,
    )
    tgt_gray = extract_single_channel(tgt_nat, tgt_chan)
    ref_gray = extract_single_channel(ref_nat, ref_chan)
    labels = np.unique(masks); labels = labels[labels > 0]
    if labels.size == 0:
        raise ValueError("No cells found in masks.")
    # For outside assignment
    fg = masks > 0
    dist_bg, idx = ndi.distance_transform_edt(~fg, return_indices=True)
    nearest_label = masks[idx[0], idx[1]]

    # Sliding windows
    s = float(start_pct); e = float(end_pct)
    wsize = float(window_size_pct); wstep = float(window_step_pct)
    if wsize <= 0: wsize = 10.0
    if wstep <= 0: wstep = 5.0
    if e < s: s, e = e, s
    
    # Generate window starts from s to e-wsize (inclusive) with step wstep
    window_starts = []
    current = s
    while current <= e - wsize + 1e-6:
        window_starts.append(current)
        current += wstep
    
    if len(window_starts) == 0:
        raise ValueError(f"Invalid window settings: no windows formed (start={s}, end={e}, size={wsize}, step={wstep})")
    
    windows = [(start, start + wsize) for start in window_starts]
    nwindows = len(windows)

    rows = []
    for lab in labels:
        cell = (masks == lab)
        if not np.any(cell):
            continue
        di = ndi.distance_transform_edt(cell)
        dmax = float(di.max())
        if dmax <= 0:
            continue
        tin = 1.0 - (di / dmax)
        out_lab = (~fg) & (nearest_label == lab)
        tout = None
        if np.any(out_lab):
            tout = np.zeros_like(tin, dtype=np.float32)
            tout[out_lab] = 1.0 + (dist_bg[out_lab].astype(np.float32) / dmax)
        for k in range(nwindows):
            a_pct, b_pct = windows[k]
            a = a_pct / 100.0
            b = b_pct / 100.0
            # Include both endpoints for the window
            idx_in = cell & (tin >= a) & (tin <= b)
            if tout is not None:
                idx_out = out_lab & (tout >= a) & (tout <= b)
                idx_band = idx_in | idx_out
            else:
                idx_band = idx_in
            n = int(np.count_nonzero(idx_band))
            if n == 0:
                rows.append({
                    "label": int(lab),
                    "band_start_pct": a_pct,
                    "band_end_pct": b_pct,
                    "center_pct": (a_pct + b_pct) / 2.0,
                    "count_px": 0,
                    "mean_target": np.nan,
                    "mean_reference": np.nan,
                    "std_target": np.nan,
                    "std_reference": np.nan,
                    "sem_target": np.nan,
                    "sem_reference": np.nan,
                    "mean_ratio_T_over_R": np.nan,
                    "std_ratio_T_over_R": np.nan,
                    "sem_ratio_T_over_R": np.nan,
                    "count_ratio_px": 0,
                })
                continue
            vals_t = tgt_gray[idx_band].astype(np.float64)
            vals_r = ref_gray[idx_band].astype(np.float64)
            mean_t = float(np.mean(vals_t))
            mean_r = float(np.mean(vals_r))
            std_t = float(np.std(vals_t, ddof=1)) if n > 1 else np.nan
            std_r = float(np.std(vals_r, ddof=1)) if n > 1 else np.nan
            sem_t = float(std_t / np.sqrt(n)) if n > 1 else np.nan
            sem_r = float(std_r / np.sqrt(n)) if n > 1 else np.nan
            r = ref_gray[idx_band].astype(np.float64)
            t = tgt_gray[idx_band].astype(np.float64)
            eps = float(ratio_ref_epsilon)
            rr = (t + eps) / (r + eps)
            ratio_mean = float(np.mean(rr)) if rr.size > 0 else np.nan
            ratio_std = float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan
            ratio_sem = float(ratio_std / np.sqrt(rr.size)) if rr.size > 1 else np.nan
            ratio_cnt = int(rr.size)
            rows.append({
                "label": int(lab),
                "band_start_pct": a_pct,
                "band_end_pct": b_pct,
                "center_pct": (a_pct + b_pct) / 2.0,
                "count_px": n,
                "mean_target": mean_t,
                "mean_reference": mean_r,
                "std_target": std_t,
                "std_reference": std_r,
                "sem_target": sem_t,
                "sem_reference": sem_r,
                "mean_ratio_T_over_R": ratio_mean,
                "std_ratio_T_over_R": ratio_std,
                "sem_ratio_T_over_R": ratio_sem,
                "count_ratio_px": ratio_cnt,
            })

    df = pd.DataFrame(rows).sort_values(["label", "center_pct"]) if rows else pd.DataFrame()
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_radial_profile_all_cells.csv")
    df.to_csv(tmp_csv.name, index=False)
    return df, tmp_csv.name

def compute_radial_peak_difference(
    radial_df: pd.DataFrame,
    quant_df: pd.DataFrame = None,
    min_pct: float = 60.0,
    max_pct: float = 120.0,
    *,
    algo: str = "global_max",
    sg_window: int = 5,
    sg_poly: int = 2,
    shoulder_slope_tol: float = 0.005,
    shoulder_curv_tol: float = 0.001,
    shoulder_plateau_span: int = 2,
    shoulder_plateau_tol: float = 0.02,
) -> pd.DataFrame:
    """
    Compute for each label the center_pct where mean_target and mean_reference reach maximum
    within the specified range, and the difference between these positions.
    Also computes distances in pixels and micrometers using equivalent radius from area.
    
    Args:
        radial_df: DataFrame from radial_profile_all_cells with columns: label, center_pct, mean_target, mean_reference
        quant_df: DataFrame from integrate_and_quantify with area_cell_px and area_cell_um2 columns (optional)
        min_pct: Minimum center_pct to consider (default: 60.0)
        max_pct: Maximum center_pct to consider (default: 120.0)
    
    Returns:
        DataFrame with columns: label, max_target_center_pct, max_reference_center_pct, difference_pct,
                                max_target_px, max_reference_px, difference_px,
                                max_target_um, max_reference_um, difference_um
    """
    if radial_df is None or radial_df.empty:
        return pd.DataFrame(columns=[
            "label", "max_target_center_pct", "max_reference_center_pct", "difference_pct",
            "max_target_px", "max_reference_px", "difference_px",
            "max_target_um", "max_reference_um", "difference_um",
            "max_target_intensity", "max_reference_intensity",
        ])
    
    # Filter by range
    df_filtered = radial_df[
        (radial_df["center_pct"] >= min_pct) & 
        (radial_df["center_pct"] <= max_pct)
    ].copy()
    
    if df_filtered.empty:
        return pd.DataFrame(columns=[
            "label", "max_target_center_pct", "max_reference_center_pct", "difference_pct",
            "max_target_px", "max_reference_px", "difference_px",
            "max_target_um", "max_reference_um", "difference_um",
            "max_target_intensity", "max_reference_intensity",
        ])
    
    # Build lookup for equivalent radius per label
    radius_px_map = {}
    radius_um_map = {}
    if quant_df is not None and not quant_df.empty:
        for _, row in quant_df.iterrows():
            try:
                lab = int(row.get("label", -1))
                if lab <= 0:
                    continue
                # Use area_cell_px and area_cell_um2 (whole cell area from segmentation)
                area_px = row.get("area_cell_px", 0.0)
                area_um2 = row.get("area_cell_um2", 0.0)
                
                # Handle potential NaN or None values
                if pd.notna(area_px) and float(area_px) > 0:
                    r_eq_px = float(np.sqrt(float(area_px) / np.pi))
                    radius_px_map[lab] = r_eq_px
                    
                if pd.notna(area_um2) and float(area_um2) > 0:
                    r_eq_um = float(np.sqrt(float(area_um2) / np.pi))
                    radius_um_map[lab] = r_eq_um
            except Exception as e:
                # Skip problematic rows
                continue
    
    labels = df_filtered["label"].unique()
    results = []
    
    def _first_local_top(series_center_pct: pd.Series, series_value: pd.Series) -> float:
        """
        Find the first local maximum when scanning from highest center_pct to lowest within range.
        If no local maximum is found, fall back to global maximum index.
        Returns the center_pct of the selected peak, or NaN if unavailable.
        """
        try:
            # Ensure sorted by center_pct ascending for consistent neighbor logic
            dfv = pd.DataFrame({
                'center_pct': series_center_pct.to_numpy(dtype=float),
                'val': series_value.to_numpy(dtype=float),
            }).dropna(subset=['center_pct', 'val']).sort_values('center_pct')
            if dfv.empty:
                return np.nan
            cp = dfv['center_pct'].to_numpy()
            v = dfv['val'].to_numpy()
            n = v.size
            # scan from right (highest %) to left, ignore endpoints for proper local maximum
            for i in range(n - 2, 0, -1):
                left = v[i - 1]
                right = v[i + 1]
                if np.isfinite(v[i]) and np.isfinite(left) and np.isfinite(right):
                    # local top (allow plateau peak as >= neighbors)
                    if v[i] >= left and v[i] >= right:
                        return float(cp[i])
            # fallback to global max
            j = int(np.nanargmax(v)) if np.any(np.isfinite(v)) else None
            return float(cp[j]) if j is not None else np.nan
        except Exception:
            return np.nan

    algo_s = str(algo or "global_max").lower()

    # --- Savitzky–Golay support and shoulder detection helpers ---
    def _prep_series_sorted(series_center_pct: pd.Series, series_value: pd.Series):
        dfv = pd.DataFrame({
            'center_pct': series_center_pct.to_numpy(dtype=float),
            'val': series_value.to_numpy(dtype=float),
        }).dropna(subset=['center_pct', 'val']).sort_values('center_pct')
        if dfv.empty:
            return None, None
        return dfv['center_pct'].to_numpy(), dfv['val'].to_numpy()

    def _savgol_nan_safe(y: np.ndarray, win: int, poly: int) -> np.ndarray:
        y = y.astype(float)
        n = y.size
        if n < 3:
            return y
        # window is odd and <= n
        win = max(3, min(win, n if n % 2 == 1 else n - 1))
        if win % 2 == 0:
            win -= 1
        if win <= poly:
            poly = max(1, min(poly, win - 1))
        m = np.isfinite(y)
        if m.sum() < 2:
            return y
        y_fill = y.copy()
        y_fill[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), y[m])
        ys = savgol_filter(y_fill, window_length=win, polyorder=poly, mode="interp")
        return ys

    def _first_shoulder(series_center_pct: pd.Series, series_value: pd.Series) -> float:
        cp, v = _prep_series_sorted(series_center_pct, series_value)
        if cp is None:
            return np.nan
        n = v.size
        if n < 5:
            return _first_local_top(series_center_pct, series_value)
        # light smoothing
        v_s = _savgol_nan_safe(v, win=int(sg_window) if n >= int(sg_window) else 3, poly=int(sg_poly))
        # normalize for relative thresholds
        v_min, v_max = float(np.nanmin(v_s)), float(np.nanmax(v_s))
        rng = max(1e-12, v_max - v_min)
        v_n = (v_s - v_min) / rng
        dv = np.gradient(v_n, cp)
        d2 = np.gradient(dv, cp)
        slope_tol = float(shoulder_slope_tol)
        curv_tol = float(shoulder_curv_tol)
        plateau_span = int(shoulder_plateau_span)
        plateau_tol = float(shoulder_plateau_tol)
        # scan from right (highest %) to left
        for i in range(n - 2, 0, -1):
            if dv[i - 1] > slope_tol and abs(dv[i]) <= slope_tol and d2[i] < -curv_tol:
                return float(cp[i])
        # plateau assistant
        for i in range(n - 2, 0, -1):
            j2 = min(n - 1, i + plateau_span)
            if dv[i - 1] > slope_tol:
                window = v_n[i:j2 + 1]
                if window.size >= 2 and (np.nanmax(window) - np.nanmin(window)) <= plateau_tol:
                    return float(cp[i])
        return _first_local_top(series_center_pct, series_value)

    for lab in labels:
        lab_data = df_filtered[df_filtered["label"] == lab]
        
        if algo_s == "first_local_top":
            # First local maximum scanning from upper bound to lower bound
            max_target_pct = _first_local_top(lab_data["center_pct"], lab_data["mean_target"])
            max_reference_pct = _first_local_top(lab_data["center_pct"], lab_data["mean_reference"])
        elif algo_s == "first_shoulder":
            max_target_pct = _first_shoulder(lab_data["center_pct"], lab_data["mean_target"])
            max_reference_pct = _first_shoulder(lab_data["center_pct"], lab_data["mean_reference"])
        else:
            # Global maximum within range (current behavior)
            valid_target = lab_data.dropna(subset=["mean_target"])
            if not valid_target.empty:
                max_target_idx = valid_target["mean_target"].idxmax()
                max_target_pct = valid_target.loc[max_target_idx, "center_pct"]
            else:
                max_target_pct = np.nan
            valid_reference = lab_data.dropna(subset=["mean_reference"])
            if not valid_reference.empty:
                max_reference_idx = valid_reference["mean_reference"].idxmax()
                max_reference_pct = valid_reference.loc[max_reference_idx, "center_pct"]
            else:
                max_reference_pct = np.nan
        
        # Get intensities at peaks
        def _get_intensity(df_lab: pd.DataFrame, pct: float, col: str) -> float:
            try:
                if not np.isfinite(pct):
                    return np.nan
                row = df_lab.loc[df_lab["center_pct"] == pct]
                if row.empty:
                    # fallback: nearest center_pct within small tolerance
                    diffs = np.abs(df_lab["center_pct"].to_numpy(dtype=float) - float(pct))
                    j = int(np.nanargmin(diffs)) if diffs.size else None
                    return float(df_lab.iloc[j][col]) if j is not None else np.nan
                return float(row.iloc[0][col])
            except Exception:
                return np.nan

        max_target_intensity = _get_intensity(lab_data, max_target_pct, "mean_target")
        max_reference_intensity = _get_intensity(lab_data, max_reference_pct, "mean_reference")

        # Calculate difference in %
        if np.isfinite(max_target_pct) and np.isfinite(max_reference_pct):
            diff_pct = max_target_pct - max_reference_pct
        else:
            diff_pct = np.nan
        
        # Convert to pixel and um using equivalent radius
        # r_eq represents the distance from 0% (center) to 100% (boundary)
        lab_int = int(lab)
        r_eq_px = radius_px_map.get(lab_int, np.nan)
        r_eq_um = radius_um_map.get(lab_int, np.nan)
        
        # Compute pixel values
        if np.isfinite(r_eq_px) and r_eq_px > 0:
            max_target_px = (max_target_pct / 100.0) * r_eq_px if np.isfinite(max_target_pct) else np.nan
            max_reference_px = (max_reference_pct / 100.0) * r_eq_px if np.isfinite(max_reference_pct) else np.nan
            diff_px = (diff_pct / 100.0) * r_eq_px if np.isfinite(diff_pct) else np.nan
        else:
            max_target_px = max_reference_px = diff_px = np.nan
        
        # Compute micrometer values
        if np.isfinite(r_eq_um) and r_eq_um > 0:
            max_target_um = (max_target_pct / 100.0) * r_eq_um if np.isfinite(max_target_pct) else np.nan
            max_reference_um = (max_reference_pct / 100.0) * r_eq_um if np.isfinite(max_reference_pct) else np.nan
            diff_um = (diff_pct / 100.0) * r_eq_um if np.isfinite(diff_pct) else np.nan
        else:
            max_target_um = max_reference_um = diff_um = np.nan
        
        results.append({
            "label": lab_int,
            "max_target_center_pct": max_target_pct,
            "max_reference_center_pct": max_reference_pct,
            "difference_pct": diff_pct,
            "max_target_px": max_target_px,
            "max_reference_px": max_reference_px,
            "difference_px": diff_px,
            "max_target_um": max_target_um,
            "max_reference_um": max_reference_um,
            "difference_um": diff_um,
            "max_target_intensity": max_target_intensity,
            "max_reference_intensity": max_reference_intensity,
        })
    
    return pd.DataFrame(results).sort_values("label")

