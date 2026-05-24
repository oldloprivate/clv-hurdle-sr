# A Two-Stage Hurdle Gradient-Boosting Framework for Zero-Inflated Customer Lifetime Value Prediction

This repository contains the full, reproducible code accompanying the paper:

> Lin, C.-Y., Chen, Y.-M., Kuo, C.-C., Yen, C.-E., & Lo, Y.-Y. *A two-stage Hurdle gradient-boosting framework for zero-inflated customer lifetime value prediction.* Scientific Reports (under review).

The framework decomposes customer lifetime value (CLV) prediction into a Stage 1 occurrence classifier (purchase probability *P*) and a Stage 2 intensity regressor (conditional spending *E*) trained on an arcsinh-transformed target, with the final prediction formed as *P* × sinh(*E*) × (1 + *S*), where *S* is Duan's smearing factor.

## Repository contents

| File | Description |
| --- | --- |
| `CLV_TwoStage_Hurdle_Pipeline_Fixed.py` | End-to-end pipeline reproducing every table and figure in the paper and its Supplementary Information. |
| `requirements.txt` | Python package dependencies. |
| `LICENSE` | MIT License. |
| `.zenodo.json` | Metadata for automatic Zenodo archiving. |

## Data

The study uses the publicly available **Online Retail II** dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Download the full dataset and provide its path to the pipeline via the `DATA_PATH` variable. The dataset is **not** redistributed in this repository.

## Environment and installation

The pipeline was developed and tested on Python 3.10+ (Google Colab). To set up locally:

```bash
pip install -r requirements.txt
```

Key dependencies: scikit-learn, xgboost, lightgbm, catboost, shap, lifetimes, pandas, numpy, scipy, matplotlib.

## Reproducing the results

1. Download `online_retail_II_full.csv` from the UCI link above.
2. Set `DATA_PATH` near the top of `CLV_TwoStage_Hurdle_Pipeline_Fixed.py` to the local path of the CSV. (If running outside Google Colab, comment out the `google.colab` import and the `drive.mount` line.)
3. Run the pipeline:

```bash
python CLV_TwoStage_Hurdle_Pipeline_Fixed.py
```

The script executes the following steps and prints every numeric result reported in the paper:

- **STEP 0–1** Invoice-level deduplication and feature engineering over a fixed 44-day prediction window (2010-11-09 to 2010-12-23).
- **STEP 2** Architecture comparison across 30 seeds (Table 1).
- **STEP 3** Dietterich 5×2 cross-validation with sign test and Bonferroni/Holm corrections (Table 2; Supplementary Table S1).
- **STEP 4** BG/NBD + Gamma-Gamma baseline and cross-model comparison (Table 3).
- **STEP 5** High-value identification (Supplementary Table S2).
- **STEP 5.5** Stage 1 standalone diagnostics and reliability diagram (Supplementary Table S3; main-text Figure 3).
- **STEP 6** Out-of-time cross-year validation (Supplementary Table S4).
- **STEP 7** Extended-feature experiment (Supplementary Table S5).
- **STEP 8** P×E segmentation, P-threshold sensitivity and profit-curve analysis (Table 4; Supplementary Tables S6–S7).
- **STEP 9** SHAP interpretability analysis (main-text Figures 1–2; Supplementary Figures S5–S10) and the smearing-factor estimate (Supplementary Table S8).

## Figure outputs

Figures are written as 600 dpi LZW-compressed TIFF files (set `OUT_FMT` to `"pdf"` for editable vector output). The naming matches the manuscript:

- Main text: `fig1_stage1_shap_bar`, `fig2_stage2_shap_bar`, `fig3_reliability_diagram`.
- Supplementary: `figS1`–`figS10`.

Files prefixed `figEx_` are additional diagnostic plots not included in the paper.

## Reproducibility notes

All experiments are seeded. The architecture comparison, statistical tests, OOT and extended-feature experiments aggregate over 30 random seeds and report mean ± standard deviation. The P-threshold sensitivity and profit-curve tables are computed on a single representative seed, as stated in the corresponding captions.

## Citation

If you use this code, please cite the paper above. A `CITATION.cff` / Zenodo DOI will be added upon publication.

## License

Released under the MIT License (see `LICENSE`).
