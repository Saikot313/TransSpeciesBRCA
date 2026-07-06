# Cross-Species Cancer Genomics Pipeline
## Dog vs Human Mammary Tumor Classification using Shared Orthologous Genes

---

## Research Question
Can a classifier trained on human TCGA-BRCA breast cancer gene expression
correctly classify canine mammary tumor malignancy using shared orthologous genes?

---

## Datasets
| Dataset | Source | Samples | Format |
|---|---|---|---|
| TCGA-BRCA (human) | UCSC Xena | ~1,100 | RNA-seq log2-FPKM |
| Canine mammary tumor | PRJNA489087 / Figshare | 154 | RNA-seq FPKM |

---

## How to Run (Google Colab)

1. Upload all 5 Python scripts + this notebook to Colab
2. Open `cancer_pipeline_colab.ipynb`
3. Run cells top to bottom
4. Total runtime: ~45-90 minutes

---

## File Structure

```
cancer_pipeline/
├── 01_download_tcga_brca.py       # Download human TCGA-BRCA
├── 02_download_canine.py          # Download canine mammary tumor data
├── 03_ortholog_mapping.py         # Ensembl BioMart ortholog mapping
├── 04_preprocessing.py            # Normalisation, filtering, PCA
├── 05_classification_ablation.py  # ML + ablation study
├── cancer_pipeline_colab.ipynb    # Master notebook
│
├── data/
│   ├── human/                     # TCGA-BRCA expression + labels
│   ├── canine/                    # Canine FPKM + metadata
│   └── aligned/                   # Shared ortholog-aligned matrices
│
└── outputs/
    ├── figures/                   # 4 thesis-quality figures
    ├── results_summary.csv        # Model comparison table
    └── ablation_results.csv       # Ablation study table
```

---

## Experiments

### A. Within-human baseline (5-fold CV)
Trains and tests on TCGA-BRCA only. Establishes expected performance ceiling.

### B. Cross-species transfer (main experiment)
Trains on full human TCGA-BRCA, predicts canine malignancy.
This answers the core research question.

### C. Ablation study
Compares: all shared genes vs top-3000 variance vs top-1000 vs top-500 vs random.
Key finding: how many orthologous genes are needed for reliable transfer?

---

## Output Figures
| Figure | Paper section |
|---|---|
| fig1_confusion_matrix.png | Results 3.2 |
| fig2_ablation.png | Results 3.3 |
| fig3_feature_importance.png | Results 3.4 / Discussion |
| fig4_model_comparison.png | Results 3.1 |
| pca_species_subtypes.png | Methods / Supplementary |

---

## Key Parameters to Tune
- `N_GENES = 3000` in step 4 (try 1000, 2000, 5000)
- `pct_id threshold = 50%` in step 3 (try 60%, 70% for stricter ortholog quality)
- `n_estimators = 300` in XGBoost (increase for better performance)

---

## Citation
- Human data: TCGA Network (2012). Nature. doi:10.1038/nature11412
- Canine data: Kim et al. (2020). Nature Communications. doi:10.1038/s41467-020-17458-0
- Ortholog mapping: Ensembl BioMart (Cunningham et al. 2022)
- XGBoost: Chen & Guestrin (2016). KDD. doi:10.1145/2939672.2939785

