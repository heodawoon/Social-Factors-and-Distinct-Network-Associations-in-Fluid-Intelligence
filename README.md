# Social factors dominate adult intelligence prediction over health and brain features

Code accompanying the manuscript by:

Da-Woon Heo†, Eunjae Kim†, Sohyun Kang, Joon-Kyung Seong, Heung-Il Suk*, and Eun Kyong Shin*

† Equal contribution; * Corresponding authors

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
git clone https://github.com/ku-milab/Social_factors_dominate_adult_intelligence_prediction_over_health_and_brain_features.git
cd Social_factors_dominate_adult_intelligence_prediction_over_health_and_brain_features

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
- knee-point based feature selection

Important scripts:
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

### Statistical analyses — `R_script/`

Contains additional statistical analyses performed in the study, including:
- logistic regression models
- average treatment effect (ATE) analysis

---

# Demo

### Demo Dataset
Due to UK Biobank data access restrictions, the original dataset cannot be 
redistributed in this repository. Researchers must obtain access through the 
official UK Biobank procedure (https://www.ukbiobank.ac.uk/).

To verify pipeline functionality without UK Biobank access, users may generate 
a synthetic dataset matching the expected column structure. The expected 
variable structure and data types are documented in the preprocessing scripts 
under `Data_process/`.

### Expected Output
When run on the actual UK Biobank dataset, the pipeline produces:
- Trained models and cross-validation predictions (saved as `.pkl` or `.pt` files)
- Performance metrics (AUC, accuracy, sensitivity, specificity) for each fold
- SHAP-based feature attribution values (mean absolute SHAP per feature)

Example expected metrics (from the manuscript, full UK Biobank cohort):
- Gf classification AUC (full feature set): 0.816
- Gf classification AUC (social features only): 0.806
- Educational attainment classification AUC: 0.743

### Expected Run Time

The benchmark below is based on **XGBoost** training (the primary model 
reported in the manuscript) on the analytic sample (N = 5,492):

**XGBoost training time (per fold)**:
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
python Step1_variable_recoding_and_renaming.py
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
5. **Statistical analyses**:
```bash
Rscript R_script/Logistic_regression_ATE.R
```

### Reproducing Manuscript Results

To reproduce the results reported in the manuscript:

1. Obtain UK Biobank access (Application ID 70034 was used for this study).
2. Place the processed UK Biobank data in the expected directory.
3. Run the full preprocessing pipeline (`Data_process/Step0` through `Step6`).
4. Run all model training scripts under `Tree_based_models/` and `DL_based_model/FT_Transformer/`.
5. Run feature attribution scripts under `Tree_based_models/Interpret/`.
6. Run statistical analyses: `Rscript R_script/Logistic_regression_ATE.R`

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

The analyses in this study were conducted under **Application ID 70034**.

---

# License

This code is released under the **MIT License**.

See the `LICENSE` file in the repository for the full license text.

---

# Citation

This manuscript is currently under peer review. Citation information will be 
updated upon publication.

If you use this code prior to publication, please cite this repository as:

```
Heo, D.-W., Kim, E., Kang, S., Seong, J.-K., Suk, H.-I., & Shin, E. K. (2026). 
Social factors dominate adult intelligence prediction over health and brain features. 
GitHub repository: https://github.com/ku-milab/Social_factors_dominate_adult_intelligence_prediction_over_health_and_brain_features
```

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
