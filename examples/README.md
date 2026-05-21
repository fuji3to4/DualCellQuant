## 📄 Paper Reproduction Workflow

A complete Jupyter Notebook is provided to reproduce the full analysis workflow from the paper **"Dual-color image analysis for quantifying fluorescence intensity in plasma membrane region of cells"** ([Analytical Sciences, 2026](https://doi.org/10.1007/s44211-026-00908-y)):

👉 **[run_reproduction_pipeline.ipynb](run_reproduction_pipeline.ipynb)** - Complete batch processing pipeline

This Notebook implements the paper's analytical workflow step-by-step:

1. **Cellpose-SAM Segmentation** - Automated cell boundary detection  
2. **Target/Reference Mask Application** - Fluorescence mask processing  
3. **Whole-cell Quantification** - Mean, sum, and ratio calculations  
4. **Radial Mask (Membrane ROI) Quantification** - EDT-normalized membrane-region-specific quantification  
5. **Radial Profile Analysis** - Distance-dependent intensity analysis from cell center to periphery  
6. **Peak Difference Analysis** - Peak position detection and profile characteristic extraction  
7. **Batch Processing** - Automated Z-stack ordering with ID maintenance  
8. **Langmuir Fitting** - Per-cell binding affinity estimation (supplementary analysis)  

### Usage

**Local Environment (VS Code / Jupyter Lab):**
```bash
jupyter notebook run_reproduction_pipeline.ipynb
```

**Run on Google Colab** (no installation required):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuji3to4/dualCellQuant/blob/main/examples/run_reproduction_pipeline.ipynb)

### Output Structure

- **Per-sample folders**: Segmentation results, masks, overlay images, and CSV tables  
- **Consolidated CSVs**: `all_samples_quantification.csv`, `all_samples_radial_mask_quantification.csv`, `all_samples_peak_differences.csv`  
- **Radial profile plots**: Per-cell profile curves with peak annotations  

### Reference Dataset

Fully reproducible using the public dataset on Zenodo ([DOI: 10.5281/zenodo.18321816](https://doi.org/10.5281/zenodo.18321816)). The Notebook automatically downloads and extracts the data.
