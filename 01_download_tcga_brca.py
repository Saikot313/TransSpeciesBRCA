# =============================================================================
# STEP 1: Download TCGA-BRCA Human Breast Cancer Dataset
# Source: UCSC Xena Browser (no login required)
# Data: RNA-seq FPKM + clinical subtypes
# =============================================================================
# Run this in Google Colab or locally with Python 3.8+
# Cell 1: Install dependencies
# !pip install pandas numpy requests tqdm

import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

os.makedirs("data/human", exist_ok=True)
os.makedirs("data/canine", exist_ok=True)
os.makedirs("data/aligned", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Directories created.")

# =============================================================================
# Cell 2: Download TCGA-BRCA FPKM expression matrix from UCSC Xena
# File: ~300MB compressed, ~1100 samples x 60498 genes
# =============================================================================

XENA_BRCA_EXPRESSION_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz"
)

XENA_BRCA_CLINICAL_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
)

XENA_BRCA_SUBTYPE_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.BRCA.sampleMap%2FBRCA_subtype"
)


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    print(f"  {desc} -> {dest_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Done: {dest_path}")


# Download expression matrix (log2 FPKM normalised by Xena)
download_file(
    XENA_BRCA_EXPRESSION_URL,
    "data/human/TCGA_BRCA_expression.tsv.gz",
    "TCGA-BRCA expression matrix"
)

# Download clinical metadata
download_file(
    XENA_BRCA_CLINICAL_URL,
    "data/human/TCGA_BRCA_clinical.tsv",
    "TCGA-BRCA clinical metadata"
)

# NOTE: A separate "BRCA_subtype" file does not reliably exist at this path on the
# Xena S3 bucket (returns 403 Forbidden). PAM50 subtypes are pulled instead from the
# clinical file's PAM50Call_RNAseq column in Cell 4 below, so no separate download needed.

# =============================================================================
# Cell 3: Load and inspect the expression matrix
# =============================================================================

print("\nLoading TCGA-BRCA expression matrix ...")
expr_human = pd.read_csv(
    "data/human/TCGA_BRCA_expression.tsv.gz",
    sep="\t",
    index_col=0,
    compression="gzip"
)

# Xena format: genes as rows, samples as columns
print(f"  Shape (genes x samples): {expr_human.shape}")
print(f"  First 5 genes: {expr_human.index[:5].tolist()}")
print(f"  First 3 sample IDs: {expr_human.columns[:3].tolist()}")

# =============================================================================
# Cell 4: Load clinical data and PAM50 subtypes
# =============================================================================

clinical = pd.read_csv("data/human/TCGA_BRCA_clinical.tsv", sep="\t", index_col=0)
print(f"\nClinical metadata shape: {clinical.shape}")
print(f"Available columns (first 10): {clinical.columns[:10].tolist()}")

# PAM50 subtypes: extracted directly from the clinical file's PAM50Call_RNAseq column
if "PAM50Call_RNAseq" in clinical.columns:
    subtypes = clinical[["PAM50Call_RNAseq"]].copy()
    subtypes.columns = ["subtype"]
    print("PAM50 subtypes extracted from clinical file.")
    print(subtypes["subtype"].value_counts())
else:
    raise KeyError(
        "PAM50Call_RNAseq column not found in clinical file. "
        f"Available columns: {clinical.columns.tolist()}"
    )

subtypes.to_csv("data/human/TCGA_BRCA_subtype_labels.csv")
print("\nSubtype labels saved.")

# =============================================================================
# Cell 5: Filter to tumor samples only (TCGA barcodes ending in -01 = primary tumor)
# =============================================================================

# TCGA sample barcodes: TCGA-XX-XXXX-01 = tumor, -11 = normal
tumor_cols = [c for c in expr_human.columns if c[13:15] == "01"]
print(f"\nTotal samples: {expr_human.shape[1]}")
print(f"Tumor samples (suffix -01): {len(tumor_cols)}")

expr_human_tumor = expr_human[tumor_cols].copy()
print(f"Expression matrix after tumor filter: {expr_human_tumor.shape}")

# Save filtered expression matrix
expr_human_tumor.to_csv("data/human/TCGA_BRCA_tumor_expression.csv")
print("Saved: data/human/TCGA_BRCA_tumor_expression.csv")
print("\nStep 1 COMPLETE.")
