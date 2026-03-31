# Semi-automated Python pipeline for analysis of fluorescent binding of peptide toxins to Kv1.1/1.2 channels

This repository contains the Python code for a semi-automated image analysis pipeline described in:

> Shmatin I.I., Nekrasova O.V., Feofanov A.V. Semi-automated processing of fluorescent images to measure the affinity of peptide blockers to potassium Kv1 channels. Bioorganic Chemistry, 2026 (in press).

The pipeline combines neural network–based membrane segmentation (Cellpose), interactive quality control of segmented cells, and automated calculation of ligand/channel fluorescence intensity ratios for quantitative analysis of fluorescent binding of scorpion peptide toxins to Kv1 channels.

## Features

- Batch processing of confocal TIFF images exported from microscope software.
- Membrane segmentation of mKate2–Kv1.1/1.2–expressing Neuro-2a cells using Cellpose.
- Interactive selection/exclusion of cells with correctly segmented plasma membranes.
- Automatic computation of per-cell and per-condition ligand/channel intensity ratios (Rav = green/red).
- Export of aggregated data tables suitable for fitting saturation, competition and kinetic binding curves (K_d, IC_50, t_1/2, t_lag).
- Transparent logging of selected/excluded objects for reproducibility.

## Repository structure

- `Main_cellpose.py` — main entry point; orchestrates batch processing of TIFF files and calls `Object_maker.py` and `cellpose_analysis.py`.
- `Object_maker.py` — automatic generation of cell masks using the Cellpose model (red channel).
- `cellpose_analysis.py` — interactive quality control of segmented cells and calculation of intensity ratios.
- `requirements.txt` — Python dependencies required to run the pipeline.
- `file_naming_notes.md` — input file and folder naming conventions and expected directory structure.
- `README.md` — this file.

(Optional)
- `example_data/` — minimal example dataset demonstrating expected input layout and example output tables.
- `LICENSE` — license for the code (e.g. MIT).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kv1-fluorescence-pipeline.git
   cd kv1-fluorescence-pipeline
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   # venv\Scripts\activate    # Windows
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For `torch` and `torchvision` with GPU support you may need to install a wheel that matches your CUDA version (see PyTorch installation instructions).

## Input data

The pipeline expects confocal images as TIFF files with a unified naming scheme and a specific folder layout. A detailed description is provided in [`file_naming_notes.md`](file_naming_notes.md). In brief:

- Images must be stored in a single folder (or subfolders) with red and green channels saved as separate files.
- Red channel: mKate2–Kv1.1/1.2 (suffix `_ch01.tif`).
- Green channel: fluorescent toxin (HgTx-GFP or Atto-HgTx) (suffix `_ch00.tif`).
- File names encode experiment ID, series, position, z-slice and channel.

Example:

- `19022025_Kv1-2_Ce4_Hg_G_5_1_z000_ch01.tif` — red channel (mKate2), series 5, position 1.
- Corresponding mask: `Object_masks/19022025_Kv1-2_Ce4_Hg_G_5_1_z000_ch01_mask.npy`.

## Usage

### Step 0: Prepare image folder

Place all TIFF files for a given experiment into a single folder (e.g. `C:\Data\Kv1-1_1-2\Ce4_vs_HgTx_G\19022025`). Make sure file names follow the convention described in `file_naming_notes.md`.

### Step 1: Generate cell masks

Run:

```bash
python Main_cellpose.py
```

The script:

- asks for the path to the folder with images,
- scans the folder for red-channel images (`_ch01.tif`),
- runs Cellpose to segment cell membranes,
- saves masks as `.npy` files in a subfolder `Object_masks`,
- prints detected image series to the console.

If `Object_masks` already exists and is non-empty, mask generation can be skipped.

### Step 2: Inspect masks

The script shows several random images with overlaid masks to allow visual inspection of segmentation quality. If masks look correct, close the window to continue.

### Step 3: Interactive cell selection and ratio calculation

Run:

```bash
python cellpose_analysis.py
```

The script:

- asks whether you are ready to start interactive processing (“Готовы начать пользовательскую обработку? (да/нет):” — answer `да` to proceed),
- lists available series and allows you to choose the order of processing,
- opens each image with its segmentation mask; each cell is colored separately,
- lets you select cells by clicking on them (inner ring turns red); clicking again deselects a cell,
- shows the number of selected cells per image and cumulatively for the current series.

After all images in a series are processed, the results are saved to CSV.

### Output

The following CSV files are generated:

- `cell_intensity_results_series_{N}.csv` — per-series data.
- `cell_intensity_results_all_series.csv` — combined data for all processed series.

Columns:

- `image` — image file name,
- `cell_id` — unique cell identifier,
- `background_red`, `background_green` — background intensities,
- `red_intensity`, `green_intensity` — mean intensities in the cell mask,
- `ratio` — green/red intensity ratio for this cell,
- `series` — series index.

You can open the final CSV in Excel, Python or R for plotting and statistical analysis (e.g. computing Rav and fitting binding curves).

## Important notes

- Use only underscores (`_`) as separators in file names; avoid spaces and special characters.
- Series numbers encoded in file names should match experimental conditions.
- If masks or images fail to load, first check that file names and folder structure follow the documented conventions.

## Citation

This code accompanies the manuscript:

Shmatin I.I., Nekrasova O.V., Feofanov A.V.
Semi-automated processing of fluorescent images to measure the affinity of peptide blockers to potassium Kv1 channels.
Bioorganic Chemistry, 2026 (manuscript in preparation / under review).

Please cite this repository as:

Shmatin I.I. et al. Kv1 fluorescence image analysis pipeline (Python code). GitHub, 2026.
URL will be provided after the repository is made public.