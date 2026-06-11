# Social factors dominate adult intelligence prediction over health and brain features

This repository provides the full pipeline used in the study, including data preprocessing, feature construction, model training, evaluation, and feature attribution analyses.

---

# System Requirements

### Operating System
- **Tested on**: Ubuntu 22.04.3 LTS
- **Kernel version**: 5.15.0-78-generic

### Hardware Environment (used for benchmarking)
- **CPU**: AMD EPYC 7513 (32 cores)
- **GPU**: NVIDIA RTX A6000 (48 GB VRAM)
- **CUDA**: 12.8
- **RAM**: 512 GB

### Minimum Recommended Hardware
- **CPU**: x86_64 architecture, 8+ cores
- **RAM**: 16 GB minimum, 32 GB recommended for the full pipeline
- **GPU**: NVIDIA GPU with CUDA 11.8+ support recommended for XGBoost GPU acceleration and FT-Transformer training. CPU-only execution is supported but training time will increase substantially.
- **Storage**: Approximately 50–100 GB of free disk space for intermediate outputs and trained models

### Software Dependencies

#### Python (3.10.18)
- xgboost == 3.1.1
- lightgbm == 4.6.0
- scikit-learn == 1.7.1
- torch == 2.8.0
- shap == 0.49.1
- captum == 0.8.0
- kneed == 0.8.5
- numpy, pandas, scipy, matplotlib
- hydra-core >= 1.3.0, omegaconf >= 2.3.0
- wandb >= 0.13.0
- tqdm

#### R (4.2.1)
- gtsummary 2.5.0
- car 3.1.5
- pROC 1.19.0.1
- grf 2.6.1

---

# Installation Guide

### Quick Install

```bash
# Python dependencies
pip install -r requirements.txt

# R dependencies (run in R console)
# install.packages(c("gtsummary", "car", "pROC", "grf"))
```

### Typical Installation Time
- Python dependencies: approximately 5–10 minutes on a standard desktop with broadband internet
- R dependencies: approximately 10–15 minutes
- **Total install time**: ~20 minutes

### Non-standard Hardware
- GPU is **not strictly required** but strongly recommended for FT-Transformer training and XGBoost GPU-accelerated training.
- All scripts include CPU fallback configurations.

---

# Repository Structure

The repository is organized into several components corresponding to the main analysis stages.

### Demo runner — `demo.py`

Self-contained script at the repository root that generates synthetic data and
runs the XGBoost pipeline end-to-end, for verifying functionality
without UK Biobank access. See the **Demo** section below.

### Data preprocessing — `Data_process/`

Scripts used to construct the analysis dataset from the UK Biobank resource.

Pipeline steps include:

- Step0: variable extraction and merging (`Step0_merge_and_extract_complete_fractional_anisotropy_data.py`)
- Step1: variable recoding (`Step1_variable_recoding_and_renaming.py`)
- Step2: disease date alignment relative to imaging visits (`Step2_redefine_disease_dates_and_code_timing_relative_to_imaging.py`)
- Step3: filtering brain-related diseases (`Step3_filter_brain_related_disease.py`)
- Step4: missing-value filtering (`Step4_filter_values.py`)
- Step5: dataset preparation for machine learning and deep learning (`Step5_re_filter_values_for_deeplearning.py`)
- Step6: cross-validation split generation (5 repetitions of 5-fold CV) (`Step6_split_5_repeat_5_fold.py`)

Key output files:
- `Step4_filter_values.py` → generates `Step4_4_binarize_disease_column.csv` (used for statistical analysis in R)
- `Step5_re_filter_values_for_deeplearning.py` → generates `Step5_refilter_categorical_for_deeplearning.csv` (used for ML and DL models)
- `Step6_split_5_repeat_5_fold.py` → generates the cross-validation splits used in all experiments

### Tree-based machine learning models — `Tree_based_models/`

Implementation of tree-based models used in the study:
- XGBoost
- LightGBM
- Random Forest

Components:
- model training pipelines
- SHAP-based feature attribution analyses
- knee-point-based feature selection

Key scripts:
- `xgboost_pipeline_shap.py`
- `random_forest_pipeline_shap.py`
- `lightgbm_pipeline_shap.py`

Feature attribution scripts are located in `Tree_based_models/Interpret/`.

### Deep learning model (FT-Transformer) — `DL_based_model/FT_Transformer/`

Implementation of the deep learning model used in this study.

Components:
- model architecture
- training pipeline
- configuration files
- utility functions

Key scripts:
- `main.py` — training entry point
- `main_interpret.py` — interpretation pipeline
- `summarize_gradient_shap.py` — attribution summarization

The implementation is based on the FT-Transformer architecture proposed by Gorishniy et al. (NeurIPS 2021).

### Regression and causal inference — `R_script/`
- Logistic regression for individual feature associations with Gf 
  (Fig. 2c, Fig. 4b, Supplementary Table 6)
- Average treatment effect (ATE) estimation via generalized random forest models for educational attainment's effect on social network properties (Fig. 4a)

Key scripts:
- `Logistic_regression.R` — logistic regression analyses
- `ATE.R` — ATE estimation via generalized random forest

### Statistical analyses — `R_script/`
Descriptive group comparisons:
- Welch's two-sample *t*-tests for continuous variables
- Pearson's chi-square tests for categorical variables
- participant characteristics tables (Supplementary Tables 10–13)

Key scripts:
- `Descriptive_table.R` — participant characteristics tables

---

# Demo

### Demo Dataset
Due to UK Biobank data access restrictions, the original dataset cannot be 
redistributed in this repository. Researchers must obtain access through the 
official UK Biobank procedure (https://www.ukbiobank.ac.uk/).

To verify pipeline functionality **without UK Biobank access**, this 
repository includes a self-contained demo script, `demo.py`, at the 
repository root. It automatically generates a small synthetic dataset whose 
column structure exactly matches the preprocessing output 
(`Step5_refilter_categorical_for_deeplearning.csv`), generates the 
cross-validation split file, and runs the XGBoost pipeline 
(`Tree_based_models/xgboost_pipeline_shap.py`) end-to-end. The pipeline 
logic is unchanged; the only edit to the original code is a one-line 
`device = 'cpu'` default so the demo runs on both CPU and GPU. No real 
data is used.

For reference, the 506 variables used in the analysis (21 social, 
438 health-related, 48 brain) are described in detail in the 
**Methods section of the manuscript** ("Variables and measures") and in 
**Supplementary Tables 10–13**, which list all features with their UK Biobank 
field numbers and coding schemes. Researchers with UK Biobank access can use 
these descriptions to construct a dataset compatible with the preprocessing 
scripts in `Data_process/`.

### Running the Demo

```bash
python demo.py                 # CPU (default, works without GPU)
python demo.py --gpu 0         # use GPU 0 if available
python demo.py --variable_type socio   # choose a feature subset
```

### Expected Output

**Demo (`demo.py`)** creates a `./demo_output/` directory:

```
demo_output/
  data/
    Step5_refilter_categorical_for_deeplearning.csv   # synthetic dataset
    Iter_5_Folds_5.json                               # CV split file
  results/
    <cls_type>/<variable_type>/<param_name>/
      XGBoost_shap_<variable_type>_final_result_value.csv
      XGBoost_shap_<variable_type>_all_iters_folds.csv
      shap_iter<it>_fold<fold>.csv                    # one per iteration/fold
    XGBoost_shap_averaged_total_validation_results.csv
```

The console prints per-fold metrics and a final averaged summary. Because 
the demo data is random synthetic noise (no real signal), the reported 
AUC/accuracy is near chance (~0.5) — this is **expected**. The demo verifies 
that the pipeline runs end-to-end and produces correctly formatted outputs, 
not that it reproduces manuscript performance (which requires real 
UK Biobank data).

**On the actual UK Biobank dataset**, the pipeline produces:
- Trained models and cross-validation predictions (saved as `.pkl` or `.pt` files)
- Performance metrics (AUC, accuracy, sensitivity, specificity) for each fold
- SHAP-based feature attribution values (mean absolute SHAP per feature)

### Expected Run Time

**Demo (`demo.py`)** with the default tiny configuration 
(600 synthetic subjects, `n_estimators=50`):
- CPU (default): under ~1 minute for the full 5×5 CV run on a normal desktop
- GPU: faster, typically a few seconds

**XGBoost training time (per fold) on the full UK Biobank dataset**:
- GPU (NVIDIA RTX A6000): ~1.08 seconds
- CPU (AMD EPYC 7513, 32 cores): ~26.69 seconds (~25× slower than GPU)

**Full XGBoost experiment** (5 repetitions × 5-fold CV × 4 feature sets = 100 folds):
- GPU: ~2 minutes
- CPU: ~45 minutes

**Other models**:
LightGBM and Random Forest require longer training times than XGBoost as they 
do not use GPU acceleration in our pipeline. The FT-Transformer, being a deep 
learning model, requires substantially longer training time and is highly 
dependent on hyperparameter configuration. Full runtime for these models, 
along with SHAP attribution and statistical analyses, can extend to several 
hours per feature set on the hardware described above.

---

# Instructions for Use

### Using Your Own Data

To apply this pipeline to a custom dataset:

1. **Prepare your data** as a CSV file with rows representing subjects and columns representing features. The column structure should match the variables described in the preprocessing scripts (see `Data_process/`).
2. **Adjust file paths** in the preprocessing and model scripts to point to your data location.
3. **Run preprocessing**:
```bash
cd Data_process
python Step0_merge_and_extract_complete_fractional_anisotropy_data.py
python Step1_variable_recoding_and_renaming.py
python Step2_redefine_disease_dates_and_code_timing_relative_to_imaging.py
python Step3_filter_brain_related_disease.py
python Step4_filter_values.py
python Step5_re_filter_values_for_deeplearning.py
python Step6_split_5_repeat_5_fold.py
```
4. **Train models**:
```bash
# Tree-based models
python Tree_based_models/xgboost_pipeline_shap.py
python Tree_based_models/lightgbm_pipeline_shap.py
python Tree_based_models/random_forest_pipeline_shap.py

# Deep learning model
python DL_based_model/FT_Transformer/main.py
```
5. **Regression, causal inference, and statistical analyses**:
```bash
Rscript R_script/Logistic_regression.R
Rscript R_script/ATE.R
Rscript R_script/Descriptive_table.R <path_to_csv> <output_dir>
```

### Reproducing Manuscript Results

To reproduce the results reported in the manuscript:

1. Obtain UK Biobank access.
2. Place the processed UK Biobank data in the expected directory.
3. Run the full preprocessing pipeline (`Data_process/Step0` through `Step6`).
4. Run all model training scripts under `Tree_based_models/` and `DL_based_model/FT_Transformer/`.
5. Run feature attribution scripts under `Tree_based_models/Interpret/`.
6. Run regression, causal inference, and statistical analyses:
   - `Rscript R_script/Logistic_regression.R`
   - `Rscript R_script/ATE.R`
   - `Rscript R_script/Descriptive_table.R <path_to_csv> <output_dir>`

All experiments use the same cross-validation splits generated by `Step6_split_5_repeat_5_fold.py` (random seed = 42) for reproducibility.

---

# External Libraries and Frameworks

This project uses the following external libraries (full version specifications listed under "System Requirements"):

### Machine Learning
- XGBoost
- LightGBM
- scikit-learn

### Deep Learning
- PyTorch

### Explainability
- SHAP
- Captum

### Utility
- kneed (knee-point based feature selection)

---

# FT-Transformer Implementation

The deep learning model in this study is based on the **FT-Transformer architecture** proposed by:

Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021.

Official implementation: https://github.com/yandex-research/rtdl-revisiting-models

The original architecture and core components were used as a reference, while the training pipeline and data integration were adapted for the multi-domain cognitive prediction tasks in this study.

---

# Data Access

This study uses data from the **UK Biobank**.

Due to data access restrictions, the dataset cannot be redistributed in this repository.

Researchers can obtain access through the official UK Biobank data access procedure: https://www.ukbiobank.ac.uk/

---

# Disclaimer

This repository is provided to support the reproducibility of the analyses 
reported in the associated research paper. The code is intended for academic 
and research purposes.

The authors make no warranties regarding the accuracy, completeness, or 
suitability of the code for any purpose other than reproducing the published 
analyses. Users are responsible for ensuring that the code is appropriate 
for their intended use and for complying with all applicable data access 
agreements.

This code does not include UK Biobank data. Users must obtain UK Biobank 
data access independently (https://www.ukbiobank.ac.uk/) and comply with 
all UK Biobank policies regarding data use and result publication.
