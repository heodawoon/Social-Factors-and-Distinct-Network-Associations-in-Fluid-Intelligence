"""
Self-contained demo for the XGBoost pipeline.

HOW TO RUN
    python demo.py                 # CPU (default, works without a GPU)
    python demo.py --gpu 0         # use CUDA GPU 0 if available

It generates a small synthetic dataset and CV-split JSON (schema-matched to
the preprocessing output; no UK Biobank data needed), then runs
Tree_based_models/xgboost_pipeline_shap.py on it. The pipeline logic is
unchanged; the only edit to the original code is a one-line `device = 'cpu'`
default so both CPU (`--gpu -1`) and GPU (`--gpu >= 0`) runs work.

EXPECTED OUTPUT
  ./demo_output/data/    -> synthetic CSV + Iter_5_Folds_5.json
  ./demo_output/results/ -> per-fold metrics, SHAP, and averaged-result CSVs
Console shows per-fold acc/auc/sen/spc and a final averaged summary.
Data is random noise, so AUC/accuracy near 0.5 is EXPECTED — the demo
verifies the pipeline runs and emits correctly-formatted outputs, not
manuscript performance (which requires real UK Biobank data).

EXPECTED RUN TIME (default: 600 subjects, n_estimators=50)
  CPU: ~1-3 min for the full 5x5 CV run.  GPU: under 1 min.
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd

# Import the pipeline's own column definitions so the synthetic data is
# schema-compatible by construction.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Tree_based_models"))
from data_utils import select_data_gf_cls, select_data_edu_cls  # noqa: E402


def build_synthetic_dataframe(cls_type, n_subjects, seed):
    """
    Create a synthetic dataframe containing every column the pipeline expects
    for variable_type='all' (the superset of all variable subsets), plus the
    'eid' and label columns.

    All feature values are random:
      - continuous columns  -> standard normal
      - categorical columns -> random integer in [0, n_categories-1]
      - disease columns     -> random binary 0/1
    """
    rng = np.random.RandomState(seed)

    if cls_type == "gf":
        category_col, continue_col, Categories = select_data_gf_cls("all")
        label_col = "fluid_2_p10"
    else:
        category_col, continue_col, Categories = select_data_edu_cls("all")
        label_col = "ed_b_2"

    data = {}
    data["eid"] = np.arange(1, n_subjects + 1, dtype=np.int64)

    y = np.zeros(n_subjects, dtype=np.int64)
    y[: n_subjects // 2] = 1
    rng.shuffle(y)
    data[label_col] = y

    for c in continue_col:
        data[c] = rng.normal(0.0, 1.0, size=n_subjects).astype(np.float32)

    # Stay within each column's declared category count so the pipeline's
    # pandas.Categorical handling behaves as it does on real data.
    for col, n_cat in zip(category_col, Categories):
        n_cat = max(int(n_cat), 1)
        data[col] = rng.randint(0, n_cat, size=n_subjects).astype(np.int64)

    return pd.DataFrame(data), label_col


def build_cv_split_json(df, label_col, n_iter, n_folds, seed):
    """
    Produce a CV-split structure identical in shape to the JSON written by
    Data_process/Step6_split_5_repeat_5_fold.py:

        {"meta": {...},
         "iterations": [
            {"iteration": 1,
             "folds": [{"fold": 1, "train_eid": [...], "test_eid": [...]}, ...]}
         ]}
    """
    from sklearn.model_selection import StratifiedKFold

    iterations = []
    eids = df["eid"].to_numpy()
    labels = df[label_col].to_numpy()

    for it in range(n_iter):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + it)
        folds = []
        for fold_id, (tr_idx, te_idx) in enumerate(skf.split(eids, labels), start=1):
            folds.append(
                {
                    "fold": fold_id,
                    "train_eid": eids[tr_idx].tolist(),
                    "test_eid": eids[te_idx].tolist(),
                }
            )
        iterations.append(
            {
                "iteration": it + 1,
                "class0_count": int((labels == 0).sum()),
                "class1_count": int((labels == 1).sum()),
                "folds": folds,
            }
        )

    return {
        "meta": {"n_iter": n_iter, "n_folds": n_folds, "seed": seed, "synthetic": True},
        "iterations": iterations,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Demo runner for the XGBoost pipeline.")
    p.add_argument("--cls_type", choices=["gf", "edu"], default="gf")
    p.add_argument(
        "--variable_type",
        choices=["all", "brain", "health", "socio",
                 "brain_health", "brain_socio", "health_socio"],
        default="socio",
        help="Which feature subset the pipeline should use for the demo.",
    )
    p.add_argument("--n_subjects", type=int, default=600,
                   help="Number of synthetic subjects (small for a fast demo).")
    p.add_argument("--n_estimators", type=int, default=50,
                   help="XGBoost boosting rounds (small for a fast demo).")
    p.add_argument("--gpu", type=int, default=-1,
                   help="CUDA device index (default: -1, forcing CPU).")
    p.add_argument("--outdir", type=str, default="./demo_output")
    return p.parse_args()


def main():
    args = parse_args()
    seed = 42

    repo_root = os.path.abspath(os.path.dirname(__file__))
    out_root = os.path.abspath(args.outdir)
    data_dir = os.path.join(out_root, "data")
    results_dir = os.path.join(out_root, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1/3 : Generating synthetic dataset (no UK Biobank data used)")
    print("=" * 70)
    df, label_col = build_synthetic_dataframe(args.cls_type, args.n_subjects, seed)
    data_csv = os.path.join(data_dir, "Step5_refilter_categorical_for_deeplearning.csv")
    df.to_csv(data_csv, index=False)
    print(f"  subjects : {len(df)}")
    print(f"  columns  : {df.shape[1]}")
    print(f"  label    : '{label_col}'  (class balance: "
          f"{int((df[label_col] == 0).sum())}/{int((df[label_col] == 1).sum())})")
    print(f"  saved to : {data_csv}")

    print()
    print("=" * 70)
    print("STEP 2/3 : Generating 5-iteration x 5-fold cross-validation split")
    print("=" * 70)
    cv = build_cv_split_json(df, label_col, n_iter=5, n_folds=5, seed=seed)
    json_path = os.path.join(data_dir, "Iter_5_Folds_5.json")
    with open(json_path, "w") as f:
        json.dump(cv, f, indent=2)
    print(f"  iterations : {len(cv['iterations'])}")
    print(f"  folds/iter : {len(cv['iterations'][0]['folds'])}")
    print(f"  saved to   : {json_path}")

    print()
    print("=" * 70)
    print("STEP 3/3 : Running the UNMODIFIED xgboost_pipeline_shap.py")
    print("=" * 70)
    cmd = [
        sys.executable,
        "xgboost_pipeline_shap.py",
        "--cls_type", args.cls_type,
        "--variable_type", args.variable_type,
        "--data_path", data_csv,
        "--json_path", json_path,
        "--outdir", results_dir,
        "--gpu", str(args.gpu),
        "--n_estimators", str(args.n_estimators),
    ]
    print("  command :", " ".join(cmd))
    print("-" * 70)
    # Run from Tree_based_models/ so the pipeline's `import data_utils` resolves.
    result = subprocess.run(cmd, cwd=os.path.join(repo_root, "Tree_based_models"))

    print("-" * 70)
    if result.returncode == 0:
        print("DEMO COMPLETED SUCCESSFULLY.")
        print(f"Results written under: {results_dir}")
        print("NOTE: data is random noise, so AUC/accuracy near 0.5 is EXPECTED.")
    else:
        print(f"DEMO FAILED (pipeline exit code {result.returncode}).")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
