# app.py
# ---
# Minimal Gradio app that uses Cellpose-SAM (cpsam) to segment a single microscopy image
# Upload an image -> run CPSAM -> return overlay and mask

import io
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import gradio as gr
from cellpose import models
import tempfile
import os

_MODEL: Optional[models.CellposeModel] = None

def get_model(use_gpu: bool = False) -> models.CellposeModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    return _MODEL

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
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
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

def colorize_masks(masks: np.ndarray, image_gray: np.ndarray) -> Image.Image:
    import scipy.ndimage as ndi
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
    return Image.fromarray(overlay)

def run_cellpose(image: Image.Image, diameter: float, flow_threshold: float, cellprob_threshold: float, chan1: int, use_gpu: bool) -> Tuple[Image.Image, Image.Image, str]:
    img_np = pil_to_numpy(image)
    gray = extract_single_channel(img_np, chan1)
    channels = [0, 0]
    model = get_model(use_gpu)
    result = model.eval(
        gray,
        diameter=None if diameter <= 0 else diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=channels,
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
        raise ValueError(f"Unexpected number of return values from model.eval: {len(result)}")
    overlay = colorize_masks(masks, gray)
    from PIL import Image as PILImage
    mask_img = PILImage.fromarray((masks.astype(np.uint32) % 256).astype(np.uint8))
    # Save masks to a temporary .npy file for download
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmp_file, masks)
    tmp_file.close()
    return overlay, mask_img, tmp_file.name

def build_ui():
    with gr.Blocks(title="Cellpose-SAM Microscopy Segmentation") as demo:
        gr.Markdown("""
        # Cellpose-SAM Segmentation
        顕微鏡画像をアップロードすると、Cellpose-SAMで細胞を検出します。
        """)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Microscopy image", sources=["upload", "clipboard"], image_mode="RGB")
                chan1 = gr.Radio(choices=[0,1,2,3], value=0, label="Channel to segment")
                diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                use_gpu = gr.Checkbox(value=False, label="Use GPU if available")
                run_btn = gr.Button("Run Cellpose-SAM")
            with gr.Column():
                overlay = gr.Image(type="pil", label="Overlay (red edges)")
                mask_img = gr.Image(type="pil", label="Mask label image")
                mask_npy = gr.File(label="Download masks (.npy)")
        run_btn.click(fn=run_cellpose, inputs=[inp, diameter, flow_th, cellprob_th, chan1, use_gpu], outputs=[overlay, mask_img, mask_npy])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
