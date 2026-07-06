# =============================================================================
# STEP 5: ML Classification + Cross-Species Evaluation + Ablation Study
# Core research question:
#   "Can a classifier trained on human TCGA-BRCA gene expression
#    correctly classify canine mammary tumor subtypes using shared
#    orthologous genes?"
#
# Experiments:
#   A. Within-human cross-validation (baseline)
#   B. Train on human -> predict canine (cross-species transfer)
#   C. Ablation: all genes vs top-variance genes vs random genes
# =============================================================================
# !pip install scikit-learn xgboost matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs/figures", exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# =============================================================================
# Cell 1: Load preprocessed data
# =============================================================================

print("Loading preprocessed data ...")
X_human  = np.load("data/aligned/X_human.npy")
y_human  = np.load("data/aligned/y_human.npy")
X_canine = np.load("data/aligned/X_canine.npy")
y_canine = np.load("data/aligned/y_canine.npy")

human_classes  = pd.read_csv("data/aligned/human_label_classes.csv", header=None)[0].tolist()
canine_classes = pd.read_csv("data/aligned/canine_label_classes.csv", header=None)[0].tolist()

print(f"X_human:  {X_human.shape}  |  Classes: {human_classes}")
print(f"X_canine: {X_canine.shape}  |  Classes: {canine_classes}")

# =============================================================================
# Cell 2: Define models
# XGBoost = primary model | RF = comparison | LR = simple baseline
# =============================================================================

MODELS = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=-1
    )
}

# =============================================================================
# Cell 3: EXPERIMENT A — Within-human 5-fold cross-validation (baseline)
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT A: Within-human 5-fold cross-validation")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_cv = {}

for name, model in MODELS.items():
    scores = cross_val_score(model, X_human, y_human, cv=cv,
                             scoring="f1_macro", n_jobs=-1)
    results_cv[name] = scores
    print(f"  {name}: F1-macro = {scores.mean():.4f} ± {scores.std():.4f}")

# =============================================================================
# Cell 4: EXPERIMENT B — Train on HUMAN, Predict CANINE (main experiment)
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT B: Train on human TCGA-BRCA, Predict on canine CMT")
print("="*60)

# Map canine binary labels to human label space for comparison
# If canine labels = benign/malignant, map to Basal/LumA as proxy
# (This mapping should be justified by literature in your paper)
# Here we use a binary human split: Basal/Her2=aggressive, rest=non-aggressive
#
# FIXED: human_classes actually contains 'Her2' (mixed case), not 'HER2'.
# This must match human_classes exactly (see label encoding printed by
# 04_preprocessing.py) or the aggressive-class samples silently get mapped
# to the wrong binary label with no error raised.
def map_to_binary(y_classes, labels):
    """Collapse multi-class to binary for cross-species comparison."""
    aggressive = {"Basal", "Her2"}
    missing = aggressive - set(labels)
    if missing:
        raise ValueError(
            f"map_to_binary(): {missing} not found in human_classes={labels}. "
            "Check the exact case/spelling of your PAM50 labels."
        )
    binary = np.array([
        1 if labels[i] in aggressive else 0
        for i in y_classes
    ])
    return binary

y_human_binary  = map_to_binary(y_human, human_classes)
y_canine_binary = y_canine  # 0=normal, 1=tumor (real labels parsed from GEO sample names)

print(f"\nHuman binary label distribution (0=non-aggressive, 1=aggressive):")
print(pd.Series(y_human_binary).value_counts())

results_transfer = {}

for name, model in MODELS.items():
    # Train on full human dataset
    model.fit(X_human, y_human_binary)

    # Predict on canine
    y_pred = model.predict(X_canine)
    y_prob = model.predict_proba(X_canine)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_canine_binary, y_pred)
    f1  = f1_score(y_canine_binary, y_pred, average="macro")
    auc = roc_auc_score(y_canine_binary, y_prob) if y_prob is not None else None

    results_transfer[name] = {
        "accuracy": acc, "f1_macro": f1, "auc": auc,
        "y_pred": y_pred, "y_prob": y_prob
    }

    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    F1-macro:  {f1:.4f}")
    if auc: print(f"    ROC-AUC:   {auc:.4f}")
    print(f"    Report:\n{classification_report(y_canine_binary, y_pred, target_names=['Normal','Tumor'], zero_division=0)}")

# =============================================================================
# Cell 5: EXPERIMENT C — Ablation Study
# Compare: all shared genes vs top-variance genes vs random gene subset
# This is your paper's key methodological contribution table
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT C: Ablation — Gene Set Comparison")
print("="*60)

best_model_name = "XGBoost"
best_model = MODELS[best_model_name]

gene_sets = {
    "All shared orthologs":      X_human.shape[1],
    "Top 3000 by variance":      3000,
    "Top 1000 by variance":      1000,
    "Top 500 by variance":       500,
    "Random 3000 genes":         3000,   # random control
}

ablation_results = {}

for label, n_genes in gene_sets.items():
    if "Random" in label:
        np.random.seed(99)
        gene_idx = np.random.choice(X_human.shape[1], n_genes, replace=False)
    else:
        n_genes = min(n_genes, X_human.shape[1])
        # Genes are already sorted by variance from step 4
        gene_idx = np.arange(n_genes)

    Xh = X_human[:, gene_idx]
    Xc = X_canine[:, gene_idx]

    # Cross-val on human
    cv_scores = cross_val_score(
        XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                      eval_metric="logloss",
                      random_state=42, n_jobs=-1),
        Xh, y_human_binary, cv=3, scoring="f1_macro"
    )

    # Transfer to canine
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                      eval_metric="logloss",
                      random_state=42, n_jobs=-1)
    m.fit(Xh, y_human_binary)
    y_pred_c = m.predict(Xc)
    f1_transfer = f1_score(y_canine_binary, y_pred_c, average="macro")

    ablation_results[label] = {
        "n_genes": len(gene_idx),
        "human_cv_f1": cv_scores.mean(),
        "human_cv_std": cv_scores.std(),
        "canine_transfer_f1": f1_transfer
    }
    print(f"  {label} (n={len(gene_idx)}): human_cv={cv_scores.mean():.3f}, transfer={f1_transfer:.3f}")

ablation_df = pd.DataFrame(ablation_results).T
ablation_df.to_csv("outputs/ablation_results.csv")
print("\nSaved: outputs/ablation_results.csv")

# =============================================================================
# Cell 6: Figures — Confusion matrix, Ablation bar chart, Feature importance
# =============================================================================

# --- Figure 1: Confusion matrix (best model on canine) ---
best_preds = results_transfer["XGBoost"]["y_pred"]
cm = confusion_matrix(y_canine_binary, best_preds)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Tumor"],
            yticklabels=["Normal", "Tumor"], ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Cross-species transfer: XGBoost\n(Trained: Human TCGA-BRCA → Tested: Canine CMT)")
plt.tight_layout()
plt.savefig("outputs/figures/fig1_confusion_matrix.png", dpi=150)
plt.close()

# --- Figure 2: Ablation study bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(ablation_df))
width = 0.35
ax.bar(x - width/2, ablation_df["human_cv_f1"], width,
       label="Human CV F1", color="#4A90D9", alpha=0.85,
       yerr=ablation_df["human_cv_std"], capsize=4)
ax.bar(x + width/2, ablation_df["canine_transfer_f1"], width,
       label="Canine Transfer F1", color="#E8704A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(ablation_df.index, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("F1-macro score")
ax.set_title("Ablation study: Gene set vs classification performance")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/figures/fig2_ablation.png", dpi=150)
plt.close()

# --- Figure 3: XGBoost Feature Importance (top 30 genes) ---
#
# FIXED: shared_ortholog_genes.csv now only has a "human_gene_symbol" column
# (no "human_ensembl_id" - see 03_ortholog_mapping.py fix). selected_genes
# is already a list of gene SYMBOLS (that's what X_human's columns are),
# so no separate ID->symbol mapping step is needed anymore - we just use
# selected_genes directly as the display labels.
MODELS["XGBoost"].fit(X_human, y_human_binary)
importances = MODELS["XGBoost"].feature_importances_
top30_idx = np.argsort(importances)[-30:][::-1]

selected_genes = pd.read_csv("data/aligned/selected_genes.csv", header=None)[0].tolist()

top30_genes = [selected_genes[i] for i in top30_idx]
top30_scores = importances[top30_idx]

fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(range(30), top30_scores[::-1], color="#4A90D9", alpha=0.85)
ax.set_yticks(range(30))
ax.set_yticklabels(top30_genes[::-1], fontsize=9)
ax.set_xlabel("Feature importance (XGBoost)")
ax.set_title("Top 30 shared orthologous genes\n(by importance for cross-species classification)")
plt.tight_layout()
plt.savefig("outputs/figures/fig3_feature_importance.png", dpi=150)
plt.close()

# --- Figure 4: Model comparison bar chart ---
model_names = list(results_cv.keys())
human_cv_means = [results_cv[m].mean() for m in model_names]
human_cv_stds  = [results_cv[m].std()  for m in model_names]
canine_f1s     = [results_transfer[m]["f1_macro"] for m in model_names]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(model_names))
ax.bar(x - 0.2, human_cv_means, 0.35, label="Human CV (5-fold)",
       color="#4A90D9", alpha=0.85, yerr=human_cv_stds, capsize=4)
ax.bar(x + 0.2, canine_f1s, 0.35, label="Canine transfer",
       color="#E8704A", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("F1-macro")
ax.set_title("Model comparison: Human CV vs Cross-species Transfer")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_model_comparison.png", dpi=150)
plt.close()

print("Saved figures:")
print("  outputs/figures/fig1_confusion_matrix.png")
print("  outputs/figures/fig2_ablation.png")
print("  outputs/figures/fig3_feature_importance.png")
print("  outputs/figures/fig4_model_comparison.png")

# =============================================================================
# Cell 7: Final results summary table
# =============================================================================

summary = []
for name in model_names:
    summary.append({
        "Model": name,
        "Human CV F1 (mean)": round(results_cv[name].mean(), 4),
        "Human CV F1 (std)":  round(results_cv[name].std(), 4),
        "Canine Transfer F1": round(results_transfer[name]["f1_macro"], 4),
        "Canine Transfer AUC": round(results_transfer[name]["auc"], 4) if results_transfer[name]["auc"] else "N/A"
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("outputs/results_summary.csv", index=False)
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))
print("\nSaved: outputs/results_summary.csv")
print("\nStep 5 COMPLETE. All experiments done.")
