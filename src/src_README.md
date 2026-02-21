# CECS 399/499 -- fNIRS Preprocessing Pipeline

## Overview

This `src/` folder contains the preprocessing pipeline used to convert
raw `.nirs` files into updated `.nirs` files containing a correctly
constructed **S matrix** derived from AUX trigger channels and the
behavioral dataset.

The pipeline:

1.  Loads raw `.nirs` files\
2.  Extracts AUX trigger onsets\
3.  Matches trials using the behavioral Excel file\
4.  Builds a new S matrix (Shape vs Color conditions)\
5.  Saves updated `.nirs` files into `data/processed_nirs/`\
6.  Logs any excluded runs

------------------------------------------------------------------------

# Folder Structure Requirements

Your project must be structured like this:

    project_root/
    │
    ├── data/
    │   ├── raw_nirs/
    │   │   ├── S1/
    │   │   ├── S2/
    │   │   └── S3/
    │   │
    │   ├── processed_nirs/
    │   │   ├── S1/
    │   │   ├── S2/
    │   │   └── S3/
    │   │
    │   └── TC_BehavioralData.xlsx
    │
    ├── src/
    │   ├── aux_to_s_matrix.py
    │   ├── batch_process_nirs.py
    │   ├── sanity_check.py
    │   └── README.md
    │
    └── .venv/

------------------------------------------------------------------------

# IMPORTANT -- Getting the Data

All `.nirs` files must be downloaded from the **OneDrive folder shared
by our mentor, Dr.Aaron Buss**.

Steps:

1.  Download the raw `.nirs` files from the shared OneDrive.
2.  Place them inside:
    -   `data/raw_nirs/S1`
    -   `data/raw_nirs/S2`
    -   `data/raw_nirs/S3`
3.  Do NOT rename the raw files.
4.  Do NOT modify raw files.

The behavioral file (`TC_BehavioralData.xlsx`) must be placed directly
inside:

    data/

------------------------------------------------------------------------

# Script Descriptions

## aux_to_s\_matrix.py

Processes a single `.nirs` file and:

-   Detects AUX trigger onsets
-   Matches them to behavioral trials
-   Constructs a 2-column S matrix:
    -   Column 0 → Shape condition
    -   Column 1 → Color condition
-   Replaces the existing `s` field
-   Saves a new `_updated.nirs` file

Safety features: - Raises an error if no AUX triggers are detected -
Raises an error if SubNum not found in behavioral file - Prevents silent
data corruption

------------------------------------------------------------------------

## batch_process_nirs.py

Processes ALL raw `.nirs` files automatically.

It: - Iterates through S1, S2, S3 folders - Normalizes subject IDs -
Calls `aux_to_s_matrix.py` - Moves processed files into: -
`data/processed_nirs/S1` - `data/processed_nirs/S2` -
`data/processed_nirs/S3` - Logs excluded runs to `excluded_runs.txt`

ID Normalization: - S1 / S2 → 3-digit ID padded with folder digit -
S3: - 4-digit ID → used directly - 3-digit ID → padded with leading 3

------------------------------------------------------------------------

# Running the Pipeline

## Step 1 -- Activate Virtual Environment

PowerShell:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

OR run directly without activation:

    .\.venv\Scripts\python.exe src\batch_process_nirs.py

## Step 2 -- Run Batch Script

From project root:

    python src\batch_process_nirs.py

------------------------------------------------------------------------

# sanity_check.py

Used for debugging individual files.

Allows: - Inspecting AUX channel ranges - Checking trigger presence -
Validating data before batch processing

------------------------------------------------------------------------

# Future Expansion

This pipeline is designed to be extensible.

Future additions may include: - Automatic trigger channel detection -
Summary statistics report generation - QC visualization scripts -
GLM-ready export scripts - Integration with MNE-NIRS

When adding new scripts: - Keep raw data untouched - Log all
exclusions - Document script purpose clearly - Update this README

------------------------------------------------------------------------

# For Group Members

If setting up the project:

1.  Clone the repository.
2.  Create a `.venv` environment.
3.  Install dependencies:

```{=html}
<!-- -->
```
    pip install numpy pandas scipy openpyxl

4.  Create folder structure as described.
5.  Download raw `.nirs` files from the shared OneDrive.
6.  Place them in the correct `data/raw_nirs` subfolders.
7.  Run the batch script from project root.

If something fails: - Check `excluded_runs.txt` - Use
`sanity_check.py` - Do NOT edit raw files
