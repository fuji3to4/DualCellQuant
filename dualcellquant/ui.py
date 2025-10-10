"""
Gradio UI for DualCellQuant.

Two modes:
1. Step-by-step Analysis: Individual control over each step (1-4)
2. Quick Radial Profile: One-click pipeline for radial analysis (1,2,3,5,6)
"""

import gradio as gr
import tempfile
import pandas as pd
import numpy as np

from dualcellquant import *

def build_ui():
    with gr.Blocks(title="DualCellQuant") as demo:
        gr.Markdown(
            """
            # 🔬 **DualCellQuant**
            *Segment, filter, and compare cells across two fluorescence channels*
            1. **Run Cellpose-SAM** to obtain segmentation masks.
            2. **Build Radial mask** (optional).
            3. **Apply Target/Reference masks**.
            4. **Integrate** (Preprocess applied only here) and view results.
            """
        )
        with gr.Tabs():
            with gr.TabItem("Dual images"):
                masks_state = gr.State()
                radial_mask_state = gr.State()
                radial_label_state = gr.State()
                tgt_mask_state = gr.State()
                ref_mask_state = gr.State()
                quant_df_state = gr.State()  # Store Step 3 quantification DataFrame (for peak analysis)
                radial_quant_df_state = gr.State()  # Store radial quantification DataFrame
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Input Images")
                        with gr.Row():
                            tgt = gr.Image(type="pil", label="Target image", image_mode="RGB", width=600)
                            ref = gr.Image(type="pil", label="Reference image", image_mode="RGB", width=600)
                        
                        with gr.Accordion("Settings", open=False):
                            reset_settings = gr.Button("Reset Settings",scale=1)
                            label_scale = gr.Slider(0.0, 5.0, value=float(LABEL_SCALE), step=0.1, label="Label size scale (0=hidden)")
                        gr.Markdown("## 1. Run Cellpose-SAM Segmentation")
                        with gr.Accordion("Segmentation params", open=True):
                            seg_source = gr.Radio(["target","reference"], value="target", label="Segment on")
                            seg_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Segmentation channel")
                            diameter = gr.Slider(0, 200, value=0, step=1, label="Diameter (px, 0=auto)")
                            flow_th = gr.Slider(0.0, 1.5, value=0.4, step=0.05, label="Flow threshold")
                            cellprob_th = gr.Slider(-6.0, 6.0, value=0.0, step=0.1, label="Cellprob threshold")
                            use_gpu = gr.Checkbox(value=True, label="Use GPU if available")           
                                
                        run_seg_btn = gr.Button("1. Run Cellpose")
                        
                        with gr.Row():
                            seg_overlay = gr.Image(type="pil", label="Segmentation overlay", width=600)
                            mask_img = gr.Image(type="pil", label="Segmentation label image", width=600)
                        seg_tiff_file = gr.File(label="Download masks (label TIFF)")
                        # Radial mask controls are moved to the bottom (after Integrate)
                        gr.Markdown("## 2. Apply Masks")
                        with gr.Accordion("Apply mask", open=True):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Target mask settings**")
                                    tgt_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Target channel")
                                    tgt_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                                    tgt_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                                    tgt_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                                    tgt_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                                with gr.Column():
                                    gr.Markdown("**Reference mask settings**")
                                    ref_chan = gr.Radio(["gray","R","G","B"], value="gray", label="Reference channel")
                                    ref_mask_mode = gr.Dropdown(["none","global_percentile","global_otsu","per_cell_percentile","per_cell_otsu"], value="global_percentile", label="Masking mode")
                                    ref_pct = gr.Slider(0.0, 100.0, value=75.0, step=1.0, label="Percentile (Top p%)")
                                    ref_sat_limit = gr.Slider(0, 255, value=254, step=1, label="Saturation limit (abs, 8-bit scale)")
                                    ref_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")
                        
                        run_tgt_btn = gr.Button("2. Apply Target & Reference masks")
                        
                        with gr.Row():
                            with gr.Column():
                                tgt_overlay = gr.Image(type="pil", label="Target mask overlay", width=600)
                                tgt_tiff = gr.File(label="Download target mask (TIFF)")

                            with gr.Column():
                                ref_overlay = gr.Image(type="pil", label="Reference mask overlay", width=600)
                                ref_tiff = gr.File(label="Download reference mask (TIFF)")
                        gr.Markdown("## 3. Integrate & Quantify")
                        with gr.Accordion("Integrate & Quantify", open=True):
                            with gr.Column():
                                pp_bg_enable = gr.Checkbox(value=False, label="Background correction")
                                pp_bg_mode = gr.Dropdown(["rolling","dark_subtract","manual"], value="dark_subtract", label="BG method")
                                pp_bg_radius = gr.Slider(1, 300, value=50, step=1, label="Rolling ball radius (px)")
                                pp_dark_pct = gr.Slider(0.0, 50.0, value=5.0, step=0.5, label="Dark percentile (%)")
                                
                                with gr.Row():
                                    bak_tar = gr.Number(value=1.0, label="Target background", scale=1)
                                    bak_ref = gr.Number(value=1.0, label="Reference background", scale=1)
                            with gr.Column():
                                pp_norm_enable = gr.Checkbox(value=False, label="Normalization")
                                pp_norm_method = gr.Dropdown([
                                    "z-score",
                                    "robust z-score",
                                    "min-max",
                                    "percentile [1,99]",
                                ], value="min-max", label="Normalization method")
                            with gr.Column():
                                ratio_eps = gr.Number(value=1e-6, label="Ratio epsilon ε (use (T+ε)/(R+ε))", scale=1)
                            with gr.Column():
                                with gr.Row():
                                    px_w = gr.Number(value=1.0, label="Pixel width (µm)", scale=1)
                                    px_h = gr.Number(value=1.0, label="Pixel height (µm)", scale=1)
                        
                        integrate_btn = gr.Button("3. Integrate & Quantify")
                        
                        with gr.Row():
                            integrate_tar_overlay = gr.Image(type="pil", label="Integrate Target overlay (AND mask)", width=600)
                            integrate_ref_overlay = gr.Image(type="pil", label="Integrate Reference overlay (AND mask)", width=600)

                        mask_tiff = gr.File(label="Download AND mask (TIFF)")
                        table = gr.Dataframe(label="Per-cell intensities & ratios", interactive=False, pinned_columns=1)
                        csv_file = gr.File(label="Download CSV")
                        with gr.Row():
                            tgt_on_and_img = gr.Image(type="pil", label="Target on AND mask", width=600)
                            ref_on_and_img = gr.Image(type="pil", label="Reference on AND mask", width=600)
                            ratio_img = gr.Image(type="pil", label="Ratio (Target/Reference) on AND mask", width=600)
                            
                        gr.Markdown("## 4. Build Radial Mask & Quantify")
                        # Radial mask section moved here (after integration)
                        with gr.Accordion("Radial mask (optional, after integration)", open=True):
                            rad_in = gr.Slider(0.0, 150.0, value=0.0, step=1.0, label="Radial inner % (0=中心)")
                            rad_out = gr.Slider(0.0, 150.0, value=100.0, step=1.0, label="Radial outer % (100=境界)")
                            rad_min_obj = gr.Slider(0, 2000, value=50, step=10, label="Remove small objects (px)")

                        run_rad_btn = gr.Button("4. Build Radial mask & Quantify")
                        # rad_overlay = gr.Image(type="pil", label="Radial mask overlay", width=600)
                        rad_overlay=gr.State()
                        
                        with gr.Row():
                            radial_tar_overlay = gr.Image(type="pil", label="Integrate Target overlay (Radial AND mask)", width=600)
                            radial_ref_overlay = gr.Image(type="pil", label="Integrate Reference overlay (Radial AND mask)", width=600)
                        rad_tiff = gr.File(label="Download radial mask (TIFF)")
                        # rad_lbl_tiff = gr.File(label="Download radial labels (label TIFF)")
                        rad_lbl_tiff=gr.State()
                        radial_table = gr.Dataframe(label="Radial per-cell intensities & ratios", interactive=False, pinned_columns=1)
                        radial_csv = gr.File(label="Download radial CSV")

                        with gr.Row():
                            radial_tgt_on_and_img = gr.Image(type="pil", label="Target on Radial AND mask", width=600)
                            radial_ref_on_and_img = gr.Image(type="pil", label="Reference on Radial AND mask", width=600)
                            radial_ratio_img = gr.Image(type="pil", label="Ratio (Target/Reference) on Radial AND mask", width=600)
                        gr.Markdown("## 5. Radial Intensity Profile")
                        # Radial profile (banded) section
                        with gr.Accordion("Radial intensity profile", open=True):
                            prof_start = gr.Number(value=0.0, label="Start %", scale=1)
                            prof_end = gr.Number(value=150.0, label="End %", scale=1)
                            prof_window_size = gr.Number(value=5.0, label="Window size (%)", scale=1)
                            prof_window_step = gr.Number(value=2.0, label="Window moving step (%)", scale=1)
                            prof_smoothing = gr.Number(value=1, label="Plot smoothing (moving avg bins)", scale=1)
                            
                        # Cache states for radial profile results (computed by 6.)
                        prof_cache_df_state = gr.State()
                        prof_cache_csv_state = gr.State()
                        prof_cache_plot_state = gr.State()
                        prof_cache_params_state = gr.State()
                        peak_diff_state = gr.State()  # Store peak difference DataFrame for plot overlay
                        
                        run_prof_btn = gr.Button("5. Compute Radial profile")
                        
                        profile_table = gr.Dataframe(label="Radial profile (all cells)", interactive=False, pinned_columns=1)
                        profile_csv = gr.File(label="Download radial profile CSV")
                        # Single-cell / All selector
                        with gr.Row():
                            with gr.Column():
                                prof_label = gr.Dropdown(choices=["All"], value="All", label="Label for single-cell profile", allow_custom_value=False)
                                prof_show_err = gr.Checkbox(value=True, label="Show error bars (SEM)")
                            run_prof_single_btn = gr.Button("Changed label, update profile",)
                        
                        profile_plot = gr.Image(type="pil", label="Radial profile plot", width=800)
                        
                        gr.Markdown("## 6. Radial Peak Difference Analysis")
                        # Peak difference section
                        with gr.Accordion("Peak difference analysis", open=True):
                            peak_min_pct = gr.Number(value=60.0, label="Min center_pct (%)", scale=1)
                            peak_max_pct = gr.Number(value=120.0, label="Max center_pct (%)", scale=1)
                        
                        run_peak_diff_btn = gr.Button("6. Compute Peak Differences")
                        
                        peak_diff_table = gr.Dataframe(label="Peak difference per label", interactive=False, pinned_columns=1)
                        peak_diff_csv = gr.File(label="Download peak difference CSV")

                # Segmentation
                def _run_seg(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu):
                    ov, seg_tif, mask_viz, masks = run_segmentation(tgt_img, ref_img, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu)
                    # Build dropdown choices for labels
                    try:
                        labs = np.unique(masks)
                        labs = labs[labs > 0]
                        choices = ["All"] + [str(int(l)) for l in labs]
                    except Exception:
                        choices = ["All"]
                    # Reset radial-profile caches upon new segmentation
                    return ov, seg_tif, mask_viz, masks, gr.update(choices=choices, value="All"), None, None, None, None
                run_seg_btn.click(
                    fn=_run_seg,
                    inputs=[tgt, ref, seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu],
                    outputs=[seg_overlay, seg_tiff_file, mask_img, masks_state, prof_label, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
                )
                # Radial mask (now at bottom)
                def _radial_and_quantify(tgt_img, ref_img, masks, rin, rout, mino, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
                    ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl = radial_mask(masks, rin, rout, mino)
                    # choose manual backgrounds only when mode is manual
                    bgm = str(bg_mode)
                    mt = float(man_t) if (bg_en and bgm == "manual") else None
                    mr = float(man_r) if (bg_en and bgm == "manual") else None
                    q_tar_ov, q_ref_ov, q_and_tiff, q_df, q_csv, q_tgt_on, q_ref_on, q_ratio = integrate_and_quantify(
                        tgt_img, ref_img, masks, tmask, rmask, tchan, rchan, pw, ph,
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=mt, manual_ref_bg=mr, roi_mask=rad_bool, roi_labels=rad_lbl, ratio_ref_epsilon=float(eps),
                    )
                    return ov, rad_bool, rad_lbl, tiff_bool, tiff_lbl, q_df, q_csv, q_tar_ov, q_ref_ov, q_tgt_on, q_ref_on, q_ratio, q_df
                run_rad_btn.click(
                    fn=_radial_and_quantify,
                    inputs=[tgt, ref, masks_state, rad_in, rad_out, rad_min_obj, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
                    outputs=[rad_overlay, radial_mask_state, radial_label_state, rad_tiff, rad_lbl_tiff, radial_table, radial_csv, radial_tar_overlay, radial_ref_overlay, radial_tgt_on_and_img, radial_ref_on_and_img, radial_ratio_img, radial_quant_df_state],
                )
                # Radial profile callback
                def _radial_profile_cb(tgt_img, ref_img, masks, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
                    # manual backgrounds only if explicitly manual mode
                    bgm = str(bg_mode)
                    mt = float(man_t) if (bg_en and bgm == "manual") else None
                    mr = float(man_r) if (bg_en and bgm == "manual") else None
                    # All-cells table + CSV
                    df_all, csv_all = radial_profile_all_cells(
                        tgt_img, ref_img, masks, tchan, rchan,
                        float(s), float(e), float(wsize), float(wstep),
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
                    )
                    # Mean plot
                    _, _, plot_img = radial_profile_analysis(
                        tgt_img, ref_img, masks, tchan, rchan,
                        float(s), float(e), float(wsize), float(wstep),
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=mt, manual_ref_bg=mr,
                        window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
                    )
                    # Build cache params signature
                    try:
                        labs = np.unique(masks)
                        labs = labs[labs > 0]
                        lab_count = int(labs.size)
                        lab_max = int(labs.max()) if lab_count > 0 else 0
                        mshape = tuple(masks.shape)
                    except Exception:
                        lab_count = 0; lab_max = 0; mshape = None
                    params = dict(
                        tchan=str(tchan), rchan=str(rchan), start=float(s), end=float(e), 
                        window_size=float(wsize), window_step=float(wstep),
                        bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_r), dark_pct=float(dark_pct),
                        norm_enable=bool(nm_en), norm_method=str(nm_m),
                        man_t=float(mt) if mt is not None else None, man_r=float(mr) if mr is not None else None,
                        ratio_eps=float(eps),
                        mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
                    )
                    return df_all, csv_all, plot_img, df_all, csv_all, plot_img, params
                run_prof_btn.click(
                    fn=_radial_profile_cb,
                    inputs=[tgt, ref, masks_state, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
                    outputs=[profile_table, profile_csv, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
                )
                def _radial_profile_single_or_all_cb(tgt_img, ref_img, masks, label_val, tchan, rchan, s, e, wsize, wstep, smoothing, show_err, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps, cache_df, cache_csv, cache_plot, cache_params, peak_df):
                    bgm = str(bg_mode)
                    mt = float(man_t) if (bg_en and bgm == "manual") else None
                    mr = float(man_r) if (bg_en and bgm == "manual") else None
                    # Build current params signature for cache matching
                    try:
                        labs_now = np.unique(masks)
                        labs_now = labs_now[labs_now > 0]
                        lab_count = int(labs_now.size)
                        lab_max = int(labs_now.max()) if lab_count > 0 else 0
                        mshape = tuple(masks.shape)
                    except Exception:
                        lab_count = 0; lab_max = 0; mshape = None
                    cur_params = dict(
                        tchan=str(tchan), rchan=str(rchan), start=float(s), end=float(e), 
                        window_size=float(wsize), window_step=float(wstep),
                        bg_enable=bool(bg_en), bg_mode=str(bg_mode), bg_radius=int(bg_r), dark_pct=float(dark_pct),
                        norm_enable=bool(nm_en), norm_method=str(nm_m),
                        man_t=float(mt) if mt is not None else None, man_r=float(mr) if mr is not None else None,
                        ratio_eps=float(eps),
                        mask_shape=mshape, lab_count=lab_count, lab_max=lab_max,
                    )
                    def params_equal(a, b):
                        try:
                            return a == b
                        except Exception:
                            return False
                    if str(label_val) == "All":
                        # Helper to rebuild All-cells mean plot from cached all-cells DF (no recompute)
                        def _build_all_plot_from_df(df_all_in: pd.DataFrame, window_bins_int: int, show_err_bool: bool, peak_df_in: pd.DataFrame = None):
                            return plot_radial_profile_with_peaks(df_all_in, peak_df_in, "All", window_bins_int, show_err_bool)

                        # If cache matches, rebuild plot from cached DF (no recompute)
                        if (cache_df is not None) and params_equal(cache_params, cur_params):
                            plot_img = _build_all_plot_from_df(cache_df, int(smoothing), bool(show_err), peak_df)
                            return cache_df, cache_csv, plot_img, cache_df, cache_csv, plot_img, cache_params
                        # Else recompute all-cells DF, then plot from DF only
                        df_all, csv_all = radial_profile_all_cells(
                            tgt_img, ref_img, masks, tchan, rchan,
                            float(s), float(e), float(wsize), float(wstep),
                            bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                            bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                            manual_tar_bg=mt, manual_ref_bg=mr, ratio_ref_epsilon=float(eps),
                        )
                        plot_img = _build_all_plot_from_df(df_all, int(smoothing), bool(show_err), peak_df)
                        return df_all, csv_all, plot_img, df_all, csv_all, plot_img, cur_params
                    else:
                        try:
                            lab = int(str(label_val))
                        except Exception:
                            lab = None
                        if lab is None:
                            return gr.update(), None, None, cache_df, cache_csv, cache_plot, cache_params
                        # Prefer cached all-cells DF for slicing if available and params match
                        use_df = None
                        if (cache_df is not None) and params_equal(cache_params, cur_params):
                            use_df = cache_df
                        else:
                            # Recompute all-cells to fill cache (so subsequent switches are instant)
                            use_df, csv_all = radial_profile_all_cells(
                                tgt_img, ref_img, masks, tchan, rchan,
                                float(s), float(e), float(wsize), float(wstep),
                                bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                                bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                                manual_tar_bg=mt, manual_ref_bg=mr,
                            )
                            # Also compute All mean plot for completeness of cache
                            _, _, cache_plot_new = radial_profile_analysis(
                                tgt_img, ref_img, masks, tchan, rchan,
                                float(s), float(e), float(wsize), float(wstep),
                                bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                                bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                                manual_tar_bg=mt, manual_ref_bg=mr,
                                window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
                            )
                            cache_df = use_df; cache_csv = csv_all; cache_plot = cache_plot_new; cache_params = cur_params
                        # Slice single label rows
                        try:
                            df1 = use_df[use_df["label"] == int(lab)].copy()
                        except Exception:
                            # Fallback to direct compute for the label
                            df1, csv1, plot1 = radial_profile_single(
                                tgt_img, ref_img, masks, lab, tchan, rchan,
                                float(s), float(e), float(wsize), float(wstep),
                                bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                                bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                                manual_tar_bg=mt, manual_ref_bg=mr,
                                window_bins=int(smoothing), show_errorbars=bool(show_err), ratio_ref_epsilon=float(eps),
                            )
                            return df1, csv1, plot1, cache_df, cache_csv, cache_plot, cache_params
                        # Write CSV for this label
                        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=f"_radial_profile_label_{lab}.csv")
                        df1.to_csv(tmp_csv.name, index=False)
                        # Plot for this label using the new helper function
                        plot1 = plot_radial_profile_with_peaks(use_df, peak_df, lab, int(smoothing), bool(show_err))
                        return df1, tmp_csv.name, plot1, cache_df, cache_csv, cache_plot, cache_params
                run_prof_single_btn.click(
                    fn=_radial_profile_single_or_all_cb,
                    inputs=[tgt, ref, masks_state, prof_label, tgt_chan, ref_chan, prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state, peak_diff_state],
                    outputs=[profile_table, profile_csv, profile_plot, prof_cache_df_state, prof_cache_csv_state, prof_cache_plot_state, prof_cache_params_state],
                )
                
                # Peak difference callback
                def _peak_diff_cb(cached_df, quant_df, min_pct, max_pct, label_val, smoothing, show_err):
                    if cached_df is None or cached_df.empty:
                        return gr.update(value=pd.DataFrame()), None, gr.update(), None
                    
                    peak_df = compute_radial_peak_difference(cached_df, quant_df, float(min_pct), float(max_pct))
                    
                    if peak_df.empty:
                        return gr.update(value=pd.DataFrame()), None, gr.update(), None
                    
                    # Save to CSV
                    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_peak_difference.csv")
                    peak_df.to_csv(tmp_csv.name, index=False)
                    tmp_csv_path = tmp_csv.name  # Extract the path string
                    tmp_csv.close()  # Close the file wrapper
                    
                    # Regenerate plot with peak markers
                    try:
                        plot_img = plot_radial_profile_with_peaks(
                            cached_df, peak_df, label_val, int(smoothing), bool(show_err), title_suffix="(with peaks)"
                        )
                    except Exception:
                        plot_img = None
                    
                    return peak_df, tmp_csv_path, plot_img, peak_df
                
                run_peak_diff_btn.click(
                    fn=_peak_diff_cb,
                    inputs=[prof_cache_df_state, quant_df_state, peak_min_pct, peak_max_pct, prof_label, prof_smoothing, prof_show_err],
                    outputs=[peak_diff_table, peak_diff_csv, profile_plot, peak_diff_state],
                )
                
                # Target/Reference masking (no ROI coupling)
                def _apply_mask_generic(img, m, ch, sat, mode, p, mino, name):
                    return apply_mask(img, m, ch, sat, mode, p, mino, None, name)
                # Combined: apply both masks in one click from the Target button
                def _apply_masks_both(tgt_img, ref_img, m, t_ch, t_sat, t_mode, t_p, t_mino, r_ch, r_sat, r_mode, r_p, r_mino):
                    t_ov, t_tiff_path, t_mask = apply_mask(tgt_img, m, t_ch, t_sat, t_mode, t_p, t_mino, None, "target_mask")
                    r_ov, r_tiff_path, r_mask = apply_mask(ref_img, m, r_ch, r_sat, r_mode, r_p, r_mino, None, "reference_mask")
                    return t_ov, t_tiff_path, t_mask, r_ov, r_tiff_path, r_mask
                run_tgt_btn.click(
                    fn=_apply_masks_both,
                    inputs=[tgt, ref, masks_state, tgt_chan, tgt_sat_limit, tgt_mask_mode, tgt_pct, tgt_min_obj, ref_chan, ref_sat_limit, ref_mask_mode, ref_pct, ref_min_obj],
                    outputs=[tgt_overlay, tgt_tiff, tgt_mask_state, ref_overlay, ref_tiff, ref_mask_state],
                )

                # Toggle percentile slider visibility based on mask mode
                def _toggle_pct_vis(mode: str):
                    m = (str(mode) if mode is not None else '').lower()
                    return gr.update(visible=(m in ("global_percentile", "per_cell_percentile")))
                tgt_mask_mode.change(
                    fn=_toggle_pct_vis,
                    inputs=[tgt_mask_mode],
                    outputs=[tgt_pct],
                )
                ref_mask_mode.change(
                    fn=_toggle_pct_vis,
                    inputs=[ref_mask_mode],
                    outputs=[ref_pct],
                )

                def _pp_bg_mode_changed_int(mode: str):
                    m = (mode or "rolling").lower()
                    return (
                        gr.update(visible=(m == "rolling")),
                        gr.update(visible=(m == "dark_subtract")),
                        gr.update(visible=(m in ("manual", "dark_subtract"))),
                        gr.update(visible=(m in ("manual", "dark_subtract"))),
                    )
                pp_bg_mode.change(
                    fn=_pp_bg_mode_changed_int,
                    inputs=[pp_bg_mode],
                    outputs=[pp_bg_radius, pp_dark_pct, bak_tar, bak_ref],
                )
                def _integrate_callback(tgt_img, ref_img, ms, tmask, rmask, tchan, rchan, pw, ph, bg_en, bg_mode, bg_r, dark_pct, nm_en, nm_m, man_t, man_r, eps):
                    # Decide manual backgrounds and display values
                    bg_mode_s = str(bg_mode)
                    out_tar_bg = man_t
                    out_ref_bg = man_r
                    if str(bg_en).lower() in ("true", "1") and bg_mode_s == "dark_subtract":
                        try:
                            out_tar_bg = compute_dark_background(tgt_img, tchan, float(dark_pct), use_native_scale=True)
                        except Exception:
                            out_tar_bg = man_t
                        try:
                            out_ref_bg = compute_dark_background(ref_img, rchan, float(dark_pct), use_native_scale=True)
                        except Exception:
                            out_ref_bg = man_r
                    # Prepare manual values to pass (only used if mode is manual)
                    man_t = float(out_tar_bg) if (bg_en and bg_mode_s == "manual") else None
                    man_r = float(out_ref_bg) if (bg_en and bg_mode_s == "manual") else None
                    res = integrate_and_quantify(
                        tgt_img, ref_img, ms, tmask, rmask, tchan, rchan,
                        pw, ph,
                        bool(bg_en), int(bg_r), bool(nm_en), nm_m,
                        bg_mode=str(bg_mode), bg_dark_pct=float(dark_pct),
                        manual_tar_bg=man_t, manual_ref_bg=man_r, ratio_ref_epsilon=float(eps),
                    )
                    # Append background values and DataFrame to outputs so UI updates
                    return (*res[:8], out_tar_bg, out_ref_bg, res[3])  # res[3] is the DataFrame
                integrate_btn.click(
                    fn=_integrate_callback,
                    inputs=[tgt, ref, masks_state, tgt_mask_state, ref_mask_state, tgt_chan, ref_chan, px_w, px_h, pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method, bak_tar, bak_ref, ratio_eps],
                    outputs=[integrate_tar_overlay, integrate_ref_overlay, mask_tiff, table, csv_file, tgt_on_and_img, ref_on_and_img, ratio_img, bak_tar, bak_ref, quant_df_state],
                )

                # ---------------- Persist settings (Dual) ----------------
                SETTINGS_KEY = "dcq_settings_v1"
                demo.load(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        tgt_chan, tgt_mask_mode, tgt_sat_limit, tgt_pct, tgt_min_obj,
                        ref_chan, ref_mask_mode, ref_sat_limit, ref_pct, ref_min_obj,
                        px_w, px_h,
                        prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err,
                        ratio_eps,
                        label_scale,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            const raw = localStorage.getItem('{SETTINGS_KEY}');
                            const d = {{
                                seg_source: 'target', seg_chan: 'gray', diameter: 0, flow_th: 0.4, cellprob_th: 0.0, use_gpu: true,
                                pp_bg_enable: false, pp_bg_mode: 'rolling', pp_bg_radius: 50, pp_dark_pct: 5.0, pp_norm_enable: false, pp_norm_method: 'z-score',
                                rad_in: 0.0, rad_out: 100.0, rad_min_obj: 50,
                                tgt_chan: 'gray', tgt_mask_mode: 'global_percentile', tgt_sat_limit: 254, tgt_pct: 75.0, tgt_min_obj: 50,
                                ref_chan: 'gray', ref_mask_mode: 'global_percentile', ref_sat_limit: 254, ref_pct: 75.0, ref_min_obj: 50,
                                px_w: 1.0, px_h: 1.0,
                                prof_start: 0.0, prof_end: 150.0, prof_window_size: 10.0, prof_window_step: 5.0, prof_smoothing: 1, prof_show_err: true,
                                ratio_eps: 1e-6,
                                label_scale: {float(LABEL_SCALE)},
                            }};
                            let s = raw ? {{...d, ...JSON.parse(raw)}} : d;
                            const mapChan = (v) => ({{0:'gray',1:'R',2:'G',3:'B'}})[v] ?? v;
                            s.seg_chan = mapChan(s.seg_chan);
                            s.tgt_chan = mapChan(s.tgt_chan);
                            s.ref_chan = mapChan(s.ref_chan);
                            return [
                                s.seg_source,
                                s.seg_chan,
                                s.diameter,
                                s.flow_th,
                                s.cellprob_th,
                                s.use_gpu,
                                s.pp_bg_enable,
                                s.pp_bg_mode,
                                s.pp_bg_radius,
                                s.pp_dark_pct,
                                s.pp_norm_enable,
                                s.pp_norm_method,
                                s.rad_in,
                                s.rad_out,
                                s.rad_min_obj,
                                s.tgt_chan,
                                s.tgt_mask_mode,
                                s.tgt_sat_limit,
                                s.tgt_pct,
                                s.tgt_min_obj,
                                s.ref_chan,
                                s.ref_mask_mode,
                                s.ref_sat_limit,
                                s.ref_pct,
                                s.ref_min_obj,
                                s.px_w,
                                s.px_h,
                                s.prof_start,
                                s.prof_end,
                                s.prof_window_size,
                                s.prof_window_step,
                                s.prof_smoothing,
                                s.prof_show_err,
                                s.ratio_eps,
                                s.label_scale,
                            ];
                        }} catch (e) {{
                            console.warn('Failed to load saved settings:', e);
                            return [
                                'target', 'gray', 0, 0.4, 0.0, true,
                                false, 'rolling', 50, 5.0, false, 'z-score',
                                0.0, 100.0, 50,
                                'gray', 'global_percentile', 254, 75.0, 50,
                                'gray', 'global_percentile', 254, 75.0, 50,
                                1.0, 1.0,
                                0.0, 150.0, 10.0, 5.0, 1, true,
                                1e-6,
                                {float(LABEL_SCALE)},
                            ];
                        }}
                    }}
                    """,
                )

                def _persist_change(comp, key: str):
                    comp.change(
                        fn=None,
                        inputs=[comp],
                        outputs=[],
                        js=f"""
                        (v) => {{
                            try {{
                                const k = '{SETTINGS_KEY}';
                                const raw = localStorage.getItem(k);
                                const s = raw ? JSON.parse(raw) : {{}};
                                s['{key}'] = v;
                                localStorage.setItem(k, JSON.stringify(s));
                            }} catch (e) {{
                                console.warn('Failed to save setting {key}:', e);
                            }}
                        }}
                        """,
                    )

                for comp, key in [
                    (seg_source, 'seg_source'), (seg_chan, 'seg_chan'), (diameter, 'diameter'), (flow_th, 'flow_th'), (cellprob_th, 'cellprob_th'), (use_gpu, 'use_gpu'),
                    (pp_bg_enable, 'pp_bg_enable'), (pp_bg_mode, 'pp_bg_mode'), (pp_bg_radius, 'pp_bg_radius'), (pp_dark_pct, 'pp_dark_pct'), (pp_norm_enable, 'pp_norm_enable'), (pp_norm_method, 'pp_norm_method'),
                    (rad_in, 'rad_in'), (rad_out, 'rad_out'), (rad_min_obj, 'rad_min_obj'),
                    (tgt_chan, 'tgt_chan'), (tgt_mask_mode, 'tgt_mask_mode'), (tgt_sat_limit, 'tgt_sat_limit'), (tgt_pct, 'tgt_pct'), (tgt_min_obj, 'tgt_min_obj'),
                    (ref_chan, 'ref_chan'), (ref_mask_mode, 'ref_mask_mode'), (ref_sat_limit, 'ref_sat_limit'), (ref_pct, 'ref_pct'), (ref_min_obj, 'ref_min_obj'),
                    (px_w, 'px_w'), (px_h, 'px_h'), (label_scale, 'label_scale'),
                    (prof_start, 'prof_start'), (prof_end, 'prof_end'), (prof_window_size, 'prof_window_size'), (prof_window_step, 'prof_window_step'), (prof_smoothing, 'prof_smoothing'), (prof_show_err, 'prof_show_err'),
                ]:
                    _persist_change(comp, key)

                def _set_label_scale(v: float):
                    global LABEL_SCALE
                    try:
                        LABEL_SCALE = float(v)
                    except Exception:
                        pass
                    return None
                label_scale.change(fn=_set_label_scale, inputs=[label_scale], outputs=[])

                reset_settings.click(
                    fn=None,
                    inputs=[],
                    outputs=[
                        seg_source, seg_chan, diameter, flow_th, cellprob_th, use_gpu,
                        pp_bg_enable, pp_bg_mode, pp_bg_radius, pp_dark_pct, pp_norm_enable, pp_norm_method,
                        rad_in, rad_out, rad_min_obj,
                        tgt_chan, tgt_mask_mode, tgt_sat_limit, tgt_pct, tgt_min_obj,
                        ref_chan, ref_mask_mode, ref_sat_limit, ref_pct, ref_min_obj,
                        px_w, px_h,
                        prof_start, prof_end, prof_window_size, prof_window_step, prof_smoothing, prof_show_err,
                        label_scale,
                    ],
                    js=f"""
                    () => {{
                        try {{
                            localStorage.removeItem('{SETTINGS_KEY}');
                        }} catch (e) {{
                            console.warn('Failed to clear settings:', e);
                        }}
                        alert('Saved settings cleared. Restoring defaults.');
                        return [
                            'target', 'gray', 0, 0.4, 0.0, true,
                            false, 'dark_subtract', 50, 5.0, false, 'min-max',
                            0.0, 100.0, 50,
                            'gray', 'none', 254, 75.0, 50,
                            'gray', 'none', 254, 75.0, 50,
                            1.0, 1.0,
                            0.0, 150.0, 10.0, 5.0, 1, true,
                            {float(LABEL_SCALE)},
                        ];
                    }}
                    """,
                )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
