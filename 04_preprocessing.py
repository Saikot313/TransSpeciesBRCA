# =============================================================================
# STEP 4: Preprocessing — Batch Correction, Feature Selection, Label Alignment
# Handles: batch effects between species, variance filtering, label encoding
# Output: X_human, y_human, X_canine, y_canine ready for ML
# =============================================================================
# !pip install scikit-learn scipy statsmodels

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# Cell 1: Load aligned matrices
# =============================================================================

print("Loading aligned expression matrices ...")
expr_human = pd.read_csv("data/aligned/human_BRCA_aligned.csv", index_col=0)
expr_canine = pd.read_csv("data/aligned/canine_CMT_aligned.csv", index_col=0)
gene_info = pd.read_csv("data/aligned/shared_ortholog_genes.csv")

print(f"  Human  (genes x samples): {expr_human.shape}")
print(f"  Canine (genes x samples): {expr_canine.shape}")

# Load labels
# CHANGED: human label is now the real tumor/normal label (from Step 1),
# not PAM50 subtype. This matches canine's real tumor/normal label, so
# Step 5 can do a genuine cancer-vs-normal cross-species task instead of
# a PAM50-subtype proxy.
tumor_normal_df = pd.read_csv("data/human/TCGA_BRCA_tumor_normal_labels.csv", index_col=0)
meta_canine = pd.read_csv("data/canine/canine_metadata_clean.csv", index_col=0)

# =============================================================================
# Cell 2: Align human sample labels to expression matrix columns
# =============================================================================

# Match TCGA sample barcodes (shorten to 15-char format if needed)
# Xena uses full barcodes; subtype file may use 12-char format
def shorten_barcode(bc, length=15):
    return bc[:length] if len(bc) >= length else bc

expr_human.columns = [shorten_barcode(c) for c in expr_human.columns]
tumor_normal_df.index = [shorten_barcode(str(i)) for i in tumor_normal_df.index]

# Find samples with both expression and label
common_human = [s for s in expr_human.columns if s in tumor_normal_df.index]
print(f"\nHuman samples with tumor/normal label: {len(common_human)} / {expr_human.shape[1]}")

expr_human_lab = expr_human[common_human].T   # -> samples x genes
y_human_raw = tumor_normal_df.loc[common_human, tumor_normal_df.columns[0]]

print(f"Tumor/Normal distribution:")
print(y_human_raw.value_counts())

# =============================================================================
# Cell 3: Align canine sample labels
# Expected label: malignancy (benign/malignant) or her2/er status
# =============================================================================

# Keep only canine samples in our expression matrix
common_canine = [s for s in expr_canine.columns if s in meta_canine.index]
print(f"\nCanine samples with metadata: {len(common_canine)} / {expr_canine.shape[1]}")

if len(common_canine) == 0:
    # NOTE: previously this silently fell back to labeling every canine
    # sample "malignant" (a single fake class), which would have broken
    # label encoding downstream and defeated the whole point of the real
    # tumor/normal labels fixed in 02_download_canine.py. Faking labels
    # here would also make the Step 5 cross-species result meaningless,
    # inconsistent with Project Contributions 1/2/4. Fail loudly instead
    # so a real metadata/ID-mismatch bug gets caught and fixed, not masked.
    raise RuntimeError(
        "No canine samples matched between expr_canine columns and "
        "meta_canine index - 0 overlap. This means the sample IDs in "
        "data/canine/canine_fpkm_clean.csv and "
        "data/canine/canine_metadata_clean.csv don't line up "
        "(check for whitespace, case, or naming differences), or Step 2 "
        "did not run correctly. Fix the ID mismatch before continuing - "
        "do NOT proceed with placeholder/fabricated labels.\n"
        f"  expr_canine.columns sample: {list(expr_canine.columns[:5])}\n"
        f"  meta_canine.index sample:   {list(meta_canine.index[:5])}"
    )

y_canine_raw = meta_canine.loc[common_canine, "malignancy"]

# Also guard against a partial-but-broken match: if every matched sample
# came back "unknown" (see infer_label_from_name in 02_download_canine.py),
# that's the same underlying problem as zero overlap and should fail the
# same way rather than silently training on a meaningless label.
if set(y_canine_raw.unique()) <= {"unknown"}:
    raise RuntimeError(
        "All matched canine samples have label 'unknown' - tumor/normal "
        "status could not be parsed from the sample names in Step 2. "
        "Check the naming convention in data/canine/canine_metadata_clean.csv "
        "before continuing."
    )

expr_canine_lab = expr_canine[common_canine].T   # -> samples x genes
print(f"Canine label distribution:\n{y_canine_raw.value_counts()}")

# =============================================================================
# Cell 4: Gene variance filtering
# Remove low-variance genes — uninformative across samples
# CHANGED: features are now selected using CANINE variance, not human.
# Rationale: per the project's contributions, canine data is the TRAINING/
# source domain and human is the target the model is evaluated on (Step 5).
# Standard ML practice selects features using training-set statistics only,
# so using canine variance here (not human) avoids leaking target-domain
# information into feature selection.
# =============================================================================

N_GENES = 3000  # Recommended: 2000-5000 for a 3-month project

gene_variances = expr_canine_lab.var(axis=0)
top_genes = gene_variances.nlargest(N_GENES).index.tolist()

print(f"\nGene variance filtering:")
print(f"  Total shared genes: {expr_canine_lab.shape[1]}")
print(f"  Keeping top {N_GENES} by variance in CANINE data (training domain)")

X_canine = expr_canine_lab[top_genes].values
X_human = expr_human_lab[top_genes].values if all(g in expr_human_lab.columns for g in top_genes) else \
          expr_human_lab[[g for g in top_genes if g in expr_human_lab.columns]].values

print(f"  X_canine shape: {X_canine.shape}")
print(f"  X_human shape: {X_human.shape}")

# =============================================================================
# Cell 5: Batch effect correction (ComBat-style z-score per dataset)
# Full ComBat requires R/pyComBat; we use per-gene z-score as simple alternative
# Each species' expression is scaled to zero mean, unit variance per gene
# =============================================================================

print("\nApplying per-gene z-score standardisation (batch normalisation) ...")

scaler_human = StandardScaler()
scaler_canine = StandardScaler()

X_human_scaled = scaler_human.fit_transform(X_human)
X_canine_scaled = scaler_canine.fit_transform(X_canine)

print(f"  Human  mean: {X_human_scaled.mean():.4f}, std: {X_human_scaled.std():.4f}")
print(f"  Canine mean: {X_canine_scaled.mean():.4f}, std: {X_canine_scaled.std():.4f}")

# =============================================================================
# Cell 6: Encode labels
# Human: tumor/normal (real cancer-vs-normal label, matches canine)
# Canine: tumor/normal (real label parsed from GEO sample names)
# =============================================================================

le_human = LabelEncoder()
y_human = le_human.fit_transform(y_human_raw.values)
print(f"\nHuman label encoding: {dict(zip(le_human.classes_, le_human.transform(le_human.classes_)))}")

le_canine = LabelEncoder()
y_canine = le_canine.fit_transform(y_canine_raw.values)
print(f"Canine label encoding: {dict(zip(le_canine.classes_, le_canine.transform(le_canine.classes_)))}")

# =============================================================================
# Cell 7: PCA visualisation — check separation and batch effects
# =============================================================================

# Combined PCA to see if species cluster separately (expected) vs mix (problem)
X_combined = np.vstack([X_human_scaled, X_canine_scaled])
species_labels = (["Human"] * len(X_human_scaled)) + (["Canine"] * len(X_canine_scaled))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_combined)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Color by species
colors_species = ["#4A90D9" if s == "Human" else "#E8704A" for s in species_labels]
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_species, alpha=0.5, s=20)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA: Human vs Canine samples")
from matplotlib.patches import Patch
axes[0].legend(handles=[
    Patch(color="#4A90D9", label="Human (TCGA-BRCA)"),
    Patch(color="#E8704A", label="Canine (CMT)")
])

# Plot 2: Human samples colored by tumor/normal status
pca_human = pca.transform(X_human_scaled)
subtype_colors = {
    "tumor": "#F44336", "normal": "#4CAF50"
}
for subtype in np.unique(y_human_raw.values):
    mask = y_human_raw.values == subtype
    axes[1].scatter(
        pca_human[mask, 0], pca_human[mask, 1],
        label=subtype, alpha=0.6, s=20,
        color=subtype_colors.get(subtype, "#888888")
    )
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_title("Human: Tumor vs Normal in PCA space")
axes[1].legend(markerscale=2, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/pca_species_subtypes.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/pca_species_subtypes.png")

# =============================================================================
# Cell 8: Save preprocessed data for ML step
# =============================================================================

np.save("data/aligned/X_human.npy", X_human_scaled)
np.save("data/aligned/y_human.npy", y_human)
np.save("data/aligned/X_canine.npy", X_canine_scaled)
np.save("data/aligned/y_canine.npy", y_canine)

# Save label encoders' classes for later use
pd.Series(le_human.classes_).to_csv("data/aligned/human_label_classes.csv", index=False)
pd.Series(le_canine.classes_).to_csv("data/aligned/canine_label_classes.csv", index=False)
pd.Series(top_genes).to_csv("data/aligned/selected_genes.csv", index=False)

print(f"\nFinal dataset summary:")
print(f"  X_human:  {X_human_scaled.shape}  |  y_human classes: {le_human.classes_}")
print(f"  X_canine: {X_canine_scaled.shape}  |  y_canine classes: {le_canine.classes_}")
print(f"  Features (shared orthologous genes): {X_human_scaled.shape[1]}")
print("\nStep 4 COMPLETE.")
