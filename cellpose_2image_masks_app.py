# app.py
# ---
# Cellpose-SAM × Gradio
# 1ch入力（グレースケール）前提。Target/Reference それぞれに
# Masking mode: none / global_otsu / global_percentile / per_cell_otsu / per_cell_percentile
# を用意。Percentileスライダは名称を統一（manual thresholdは撤廃）。
# ANDマスク = (Target条件) ∧ (Reference条件) ※ none は画像フィルタを作らず、非サチュのみ適用。
# 可視化: 赤=Cellpose境界, 緑=ANDマスク境界, 重心にセルID。

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import tempfile
from skimage import filters, morphology, measure, color
import scipy.ndimage as ndi
from cellpose import models

# -----------------------
# Model lazy loader
# -----------------------
_MODEL: Optional[models.CellposeModel] = None

def get_model(use_gpu: bool = False) -> models.CellposeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    return _MODEL

# -----------------------
# Image helpers
# -----------------------

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3].astype(np.float32)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def extract_single_channel(img: np.ndarray, chan: int) -> np.ndarray:
    if img.ndim == 2:
        return img
    if chan == 0:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    elif chan in (1, 2, 3):
        return img[:, :, chan - 1].astype(np.float32)
    else:
        raise ValueError("Invalid channel selection")

# -----------------------
# Visualization
# -----------------------

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
    draw = ImageDraw.Draw(img)
    w, h = img.size
    fsize = max(10, int(min(w, h) * 0.02))
    try:
        font = ImageFont.truetype("arial.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    props = measure.regionprops(masks)
    for p in props:
        lab = p.label
        cy, cx = p.centroid
        x, y = int(cx), int(cy)
        text = str(lab)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((x+dx, y+dy), text, fill=(255,255,255), font=font, anchor="mm")
        draw.text((x, y), text, fill=(0,0,0), font=font, anchor="mm")
    return img

# -----------------------
# Threshold utilities
# -----------------------

def global_threshold_mask(img_gray: np.ndarray, sat_limit: float, mode: str, pct: float, min_obj_size: int) -> Tuple[np.ndarray, float]:
    """Global Otsu/Percentile with saturation exclusion and cleanup. Returns (mask, threshold_value)."""
    nonsat = img_gray < sat_limit
    if mode == "global_otsu":
        valid = img_gray[nonsat]
        if valid.size == 0:
            raise ValueError("All pixels are saturated; lower the saturation limit")
        th = float(filters.threshold_otsu(valid))
    elif mode == "global_percentile":
        valid = img_gray[nonsat]
        if valid.size == 0:
            raise ValueError("All pixels are saturated; lower the saturation limit")
        th = float(np.percentile(valid, float(np.clip(pct, 0.0, 100.0))))
    else:
        raise ValueError("global_threshold_mask called with invalid mode")
    mask = (img_gray >= th) & nonsat
    if min_obj_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_obj_size))
        mask = morphology.binary_opening(mask, morphology.disk(1))
    return mask, th


def per_cell_threshold(cell_pixels: np.ndarray, mode: str, pct: float) -> float:
    if cell_pixels.size == 0:
        return np.nan
    if mode == "per_cell_otsu":
        return float(filters.threshold_otsu(cell_pixels))
    elif mode == "per_cell_percentile":
        return float(np.percentile(cell_pixels, float(np.clip(pct, 0.0, 100.0))))
    else:
        raise ValueError("per_cell_threshold called with invalid mode")


def cleanup_mask(mask: np.ndarray, min_obj_size: int) -> np.ndarray:
    if min_obj_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_obj_size))
        mask = morphology.binary_opening(mask, morphology.disk(1))
    return mask

# -----------------------
# Core: segmentation + quantification
# -----------------------

def segment_and_quantify(
    target_img: Image.Image,
    reference_img: Image.Image,
    seg_source: str,
    seg_channel: int,
    tgt_measure_channel: int,
    ref_measure_channel: int,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    # Target (membrane)
    tgt_sat_limit: float,
    tgt_mask_mode: str,            # none | global_otsu | global_percentile | per_cell_otsu | per_cell_percentile
    tgt_pct: float,
    tgt_min_obj_size: int,
    # Reference
    ref_sat_limit: float,
    ref_mask_mode: str,
    ref_pct: float,
    ref_min_obj_size: int,
    use_gpu: bool,
):
    # inputs
    tgt = pil_to_numpy(target_img)
    ref = pil_to_numpy(reference_img)
    if tgt.shape[:2] != ref.shape[:2]:
        raise ValueError(f"Image size mismatch: target {tgt.shape[:2]} vs reference {ref.shape[:2]}")

    seg_arr_rgb = tgt if seg_source == "target" else ref
    seg_gray = extract_single_channel(seg_arr_rgb, seg_channel)

    model = get_model(use_gpu)
    result = model.eval(
        seg_gray,
        diameter=None if diameter <= 0 else diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0],  # single-channel path
        normalize=True,
        invert=False,
        compute_masks=True,
        progress=None,
    )
    if isinstance(result, tuple) and len(result) == 4:
        masks, flows, styles, diams = result
    elif isinstance(result, tuple) and len(result) == 3:
        masks, flows, styles = result
        diams = None
    else:
        raise ValueError("Unexpected return values from model.eval")

    # channels for measurement
    tgt_gray = extract_single_channel(tgt, tgt_measure_channel)
    ref_gray = extract_single_channel(ref, ref_measure_channel)
    tgt_nonsat = tgt_gray < float(tgt_sat_limit)
    ref_nonsat = ref_gray < float(ref_sat_limit)

    # Precompute global masks if needed
    tgt_global_mask = None; tgt_global_th = np.nan
    if tgt_mask_mode in ("global_otsu", "global_percentile"):
        tgt_global_mask, tgt_global_th = global_threshold_mask(tgt_gray, float(tgt_sat_limit), tgt_mask_mode, float(tgt_pct), int(tgt_min_obj_size))
    ref_global_mask = None; ref_global_th = np.nan
    if ref_mask_mode in ("global_otsu", "global_percentile"):
        ref_global_mask, ref_global_th = global_threshold_mask(ref_gray, float(ref_sat_limit), ref_mask_mode, float(ref_pct), int(ref_min_obj_size))

    # visualization union mask (for green edges)
    vis_union = np.zeros_like(masks, dtype=bool)

    labels = np.unique(masks); labels = labels[labels > 0]
    rows = []

    for lab in labels:
        cell = masks == lab
        # Target per-cell logical
        if tgt_mask_mode == "none":
            tgt_mask_cell = cell
            tgt_th_used = np.nan
        elif tgt_mask_mode in ("global_otsu", "global_percentile"):
            tgt_mask_cell = tgt_global_mask & cell
            tgt_th_used = tgt_global_th
            tgt_mask_cell = cleanup_mask(tgt_mask_cell, int(tgt_min_obj_size))  # per-cell cleanup too
        elif tgt_mask_mode in ("per_cell_otsu", "per_cell_percentile"):
            pool = tgt_gray[cell & tgt_nonsat]
            th = per_cell_threshold(pool, tgt_mask_mode, float(tgt_pct))
            base = (tgt_gray >= th) & tgt_nonsat & cell if not np.isnan(th) else np.zeros_like(cell, dtype=bool)
            tgt_mask_cell = cleanup_mask(base, int(tgt_min_obj_size))
            tgt_th_used = th
        else:
            raise ValueError("Invalid tgt_mask_mode")

        # Reference per-cell logical
        if ref_mask_mode == "none":
            ref_mask_cell = cell
            ref_th_used = np.nan
        elif ref_mask_mode in ("global_otsu", "global_percentile"):
            ref_mask_cell = ref_global_mask & cell
            ref_th_used = ref_global_th
            ref_mask_cell = cleanup_mask(ref_mask_cell, int(ref_min_obj_size))
        elif ref_mask_mode in ("per_cell_otsu", "per_cell_percentile"):
            pool = ref_gray[cell & ref_nonsat]
            th = per_cell_threshold(pool, ref_mask_mode, float(ref_pct))
            base = (ref_gray >= th) & ref_nonsat & cell if not np.isnan(th) else np.zeros_like(cell, dtype=bool)
            ref_mask_cell = cleanup_mask(base, int(ref_min_obj_size))
            ref_th_used = th
        else:
            raise ValueError("Invalid ref_mask_mode")

        idx = tgt_mask_cell & ref_mask_cell
        vis_union |= idx

        area_cell = int(cell.sum())
        area_and = int(idx.sum())
        if area_and == 0:
            mean_t_mem = np.nan; sum_t_mem = np.nan
            mean_r_mem = np.nan; sum_r_mem = np.nan
            ratio_mean = np.nan; ratio_sum = np.nan
        else:
            mean_t_mem = float(np.mean(tgt_gray[idx]))
            sum_t_mem = float(np.sum(tgt_gray[idx]))
            mean_r_mem = float(np.mean(ref_gray[idx]))
            sum_r_mem = float(np.sum(ref_gray[idx]))
            ratio_mean = float(np.mean(tgt_gray[idx]/ref_gray[idx])) if np.all(ref_gray[idx] > 0) else np.nan
            ratio_std = float(np.std(tgt_gray[idx]/ref_gray[idx])) if np.all(ref_gray[idx] > 0) else np.nan
            ratio_sum  = float(np.sum(tgt_gray[idx]/ref_gray[idx])) if np.all(ref_gray[idx] > 0) else np.nan

        # whole-cell context
        mean_t_whole = float(np.mean(tgt_gray[cell] ))
        sum_t_whole  = float(np.sum (tgt_gray[cell] ))
        mean_r_whole = float(np.mean(ref_gray[cell] ))
        sum_r_whole  = float(np.sum (ref_gray[cell] ))

        rows.append({
            "label": int(lab),
            "area_cell_px": area_cell,
            "area_and_px": area_and,
            "mean_target_on_mask": mean_t_mem,
            "sum_target_on_mask": sum_t_mem,
            "mean_reference_on_mask": mean_r_mem,
            "sum_reference_on_mask": sum_r_mem,
            "ratio_sum_T_over_R": ratio_sum,
            "ratio_mean_T_over_R": ratio_mean,
            "ratio_std_T_over_R": ratio_std,
            "mean_target_whole": mean_t_whole,
            "sum_target_whole": sum_t_whole,
            "mean_reference_whole": mean_r_whole,
            "sum_reference_whole": sum_r_whole,
            "tgt_mask_mode": tgt_mask_mode,
            "tgt_threshold_used": float(tgt_th_used) if not np.isnan(tgt_th_used) else np.nan,
            "tgt_pct": float(tgt_pct),
            "ref_mask_mode": ref_mask_mode,
            "ref_threshold_used": float(ref_th_used) if not np.isnan(ref_th_used) else np.nan,
            "ref_pct": float(ref_pct),
            "tgt_sat_limit": float(tgt_sat_limit),
            "ref_sat_limit": float(ref_sat_limit),
        })

    df = pd.DataFrame(rows).sort_values("label")

    # artifacts
    tmp_npy = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_npy, masks)
    tmp_npy.flush(); tmp_npy.close()

    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp_csv.name, index=False)

    # images
    overlay = colorize_overlay(seg_gray, masks, vis_union)
    overlay = annotate_ids(overlay, masks)
    mask_viz = vivid_label_image(masks)

    return overlay, mask_viz, tmp_npy.name, df, tmp_csv.name

# -----------------------
# UI
# -----------------------

def build_ui():
    with gr.Blocks(title="Cellpose-SAM: 2ch Quant (Global / Per-cell AND)") as demo:
        gr.Markdown(
            """
            # Cellpose-SAM: 2-channel Quantification
            - 入力は **1チャンネル（グレースケール）** を想定。
            - Masking mode は **none / global_otsu / global_percentile / per_cell_otsu / per_cell_percentile**。
            - Percentile スライダで **Top p%** を指定（global/per-cell両方で使用）。
            - **赤**=Cellpose境界, **緑**=ANDマスク境界, **番号**=セルID。
            """
        )
        with gr.Row():
            with gr.Column():
                tgt = gr.Image(type="pil", label="Target image (membrane)", image_mode="L")
                ref = gr.Image(type="pil", label="Reference image (cytosol)", image_mode="L")
                
                with gr.Accordion("Channels (optional)", open=False):
                    seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                    seg_chan = gr.Radio([0,1,2,3], value=0, label="Segmentation channel (0=gray,1=R,2=G,3=B)")
                    tgt_chan = gr.Radio([0,1,2,3], value=0, label="Target(measure) channel")
                    ref_chan = gr.Radio([0,1,2,3], value=0, label="Reference(measure) channel")

                with gr.Accordion("Cellpose params", open=False):
                    diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                    flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                    cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")

                with gr.Accordion("Target (membrane) mask", open=True):
                    tgt_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    tgt_sat_limit = gr.Slider(0.80, 1.0, value=0.98, step=0.001, label="Saturation limit (Target<limit)")
                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    tgt_min_obj_size = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")

                with gr.Accordion("Reference mask", open=True):
                    ref_mask_mode = gr.Radio(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                    ref_sat_limit = gr.Slider(0.80, 1.0, value=0.98, step=0.001, label="Saturation limit (Reference<limit)")
                    ref_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                    ref_min_obj_size = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")


                use_gpu = gr.Checkbox(value=False, label="Use GPU if available")
                run_btn = gr.Button("Run")
            with gr.Column():
                overlay = gr.Image(type="pil", label="Overlay (IDs on): red=cell edges, green=AND edges")
                mask_img = gr.Image(type="pil", label="Mask label image (colored)")
                mask_npy = gr.File(label="Download masks (.npy)")
                table = gr.Dataframe(label="Per-cell intensities & ratios (AND mask)", interactive=False)
                csv_file = gr.File(label="Download CSV")

        run_btn.click(
            fn=segment_and_quantify,
            inputs=[
                tgt, ref,
                seg_source, seg_chan, tgt_chan, ref_chan,
                diameter, flow_th, cellprob_th,
                tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj_size,
                ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj_size,
                use_gpu,
            ],
            outputs=[overlay, mask_img, mask_npy, table, csv_file],
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
