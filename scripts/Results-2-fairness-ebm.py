#!/usr/bin/env python3
"""
Fairness Testing: Base vs CSRI Model Comparison with EBM
=========================================================
This script performs comprehensive statistical testing comparing baseline 
and CSRI models using Explainable Boosting Machine (EBM) with:

1. Repeated 5-fold cross-validation (3 repetitions = 15 folds total)
2. Paired t-tests for statistical significance
3. Bootstrap confidence intervals
4. EBM explainability analysis

Expected files:
- base_exact_2031.csv
- csfri_exact_2031.csv

Outputs:
- Statistical significance results
- Bootstrap confidence intervals
- EBM global feature importances (CSV and visualisation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION - MUST MATCH THESIS EXACTLY
# ==============================================================================

# CSV file paths (assumed to be in same directory as script)
BASE_PATH = "base_exact_2031.csv"
CSRI_PATH = "csfri_exact_2031.csv"

# Random state for reproducibility
RANDOM_STATE = 42

# Number of bootstrap iterations
N_BOOTSTRAP = 1000

# Top features to show in explainability
TOP_GLOBAL = 12
TOP_LOCAL_FEATURES = 5
TOP_AT_RISK_CASES = 5

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def prepare_xy(df: pd.DataFrame):
    """Prepare data for EBM (with one-hot encoding)"""
    df = df.assign(
        Current_Year=df['Current_NC_Year'].fillna(0) if 'Current_NC_Year' in df.columns else 0,
        Pupil_Premium_Indicator=df['Pupil_Premium_Indicator'].fillna("Unknown") if 'Pupil_Premium_Indicator' in df.columns else "Unknown",
        SEN_Status=df['SEN_Status'].fillna("Unknown") if 'SEN_Status' in df.columns else "Unknown",
    )
    for col in ['Gender', 'Pupil_Premium_Indicator', 'SEN_Status']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col])
    features = df.columns.difference(['Is_CS_Incident', 'studentkey'])
    X = df[features].copy()
    y = df['Is_CS_Incident'].astype(int).copy()
    return X, y


def load_xy(path: str):
    """Load and prepare data from CSV"""
    df = pd.read_csv(path)
    return prepare_xy(df)


def scores_on_given_splits(X, y, splits, model_params=None, tune_for_precision=False):
    """Calculate scores with optional hyperparameter tuning"""
    accs, precs, recs, f1s = [], [], [], []
    
    for fold_idx, (tr, te) in enumerate(splits):
        if tune_for_precision and model_params is None:
            # Grid search for best precision
            param_grid = {
                'max_rounds': [5000, 10000],
                'learning_rate': [0.01, 0.005],
                'min_samples_leaf': [2, 5, 10],
                'max_leaves': [3, 5],
            }
            
            clf = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
            grid = GridSearchCV(clf, param_grid, cv=3, scoring='precision', n_jobs=-1)
            grid.fit(X.iloc[tr], y.iloc[tr])
            clf = grid.best_estimator_
            print(f"Fold {fold_idx+1} best params: {grid.best_params_}")
        else:
            # Use provided params or defaults
            if model_params:
                clf = ExplainableBoostingClassifier(random_state=RANDOM_STATE, **model_params)
            else:
                clf = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
            clf.fit(X.iloc[tr], y.iloc[tr])
        
        yhat = clf.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], yhat))
        precs.append(precision_score(y.iloc[te], yhat, zero_division=0))
        recs.append(recall_score(y.iloc[te], yhat, zero_division=0))
        f1s.append(f1_score(y.iloc[te], yhat, zero_division=0))
    
    return {
        "accuracy": np.array(accs),
        "precision": np.array(precs),
        "recall": np.array(recs),
        "f1": np.array(f1s),
    }


def paired_t_test_with_details(base_arr, csri_arr, metric_name):
    """Perform paired t-test with detailed output"""
    t, p = ttest_rel(base_arr, csri_arr)
    diff = csri_arr - base_arr
    
    print(f"\n{metric_name.upper()} Analysis:")
    print(f"  Base scores by fold: {base_arr.round(4)}")
    print(f"  CSRI scores by fold: {csri_arr.round(4)}")
    print(f"  Differences by fold: {diff.round(4)}")
    print(f"  Mean difference: {diff.mean():.4f} ± {diff.std():.4f}")
    print(f"  T-statistic: {t:.4f}, p-value: {p:.6f}")
    
    return float(t), float(p)


def bootstrap_ci(base_scores, csri_scores, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence interval for the difference"""
    n = len(base_scores)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_diff = csri_scores[idx].mean() - base_scores[idx].mean()
        diffs.append(boot_diff)
    
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, alpha/2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha/2) * 100)
    
    return diffs.mean(), ci_lower, ci_upper


def prettify_name(name: str) -> str:
    """Convert feature names to readable format"""
    n = str(name).replace('_', ' ')
    n = n.replace('Pupil Premium Indicator', 'Pupil Premium')
    n = n.replace('SEN Status', 'SEN status')
    n = n.replace('csfri', 'CSRI').replace('Csri', 'CSRI')
    # Handle one-hot style names like "Gender_Male"
    if '_' in str(name):
        parts = str(name).split('_', 1)
        if len(parts) == 2:
            base, cat = parts
            base_readable = base.replace('_', ' ')
            return f"{base_readable}: {cat}"
    return n.title()


def extract_global_importance(ebm: ExplainableBoostingClassifier):
    """Extract global feature importances from EBM"""
    gexp = ebm.explain_global()
    gdata = gexp.data()
    names = gdata.get("names", []) or gdata.get("feature_names", [])
    scores = gdata.get("scores", []) or gdata.get("importance", [])
    gdf = pd.DataFrame({"Feature": names, "Importance": scores})
    # Drop interaction terms for simplicity
    gdf = gdf[~gdf["Feature"].astype(str).str.contains(" x ")].copy()
    gdf["Feature (readable)"] = gdf["Feature"].astype(str).apply(prettify_name)
    gdf = gdf.sort_values("Importance", ascending=False).reset_index(drop=True)
    return gdf


def try_local_explanations(ebm: ExplainableBoostingClassifier, X_cases: pd.DataFrame):
    """
    Try to get local reason codes via explain_local. Returns a list of dicts:
      [{"names":[...], "scores":[...]}, ...]  or None if not available in this version.
    """
    try:
        lexp = ebm.explain_local(X_cases)
        ldata = lexp.data()
        # Case A: top-level lists
        if ldata and isinstance(ldata.get("names"), list) and isinstance(ldata.get("scores"), list):
            return [{"names": ldata["names"][i], "scores": ldata["scores"][i]} for i in range(len(X_cases))]
        # Case B: nested under 'specific'
        spec = ldata.get("specific") if isinstance(ldata, dict) else None
        if isinstance(spec, list) and len(spec) == len(X_cases):
            out = []
            for i in range(len(X_cases)):
                row = spec[i] if isinstance(spec[i], dict) else {}
                names = row.get("names") or row.get("feature_names") or row.get("terms") or []
                scores = row.get("scores") or row.get("values") or row.get("contrib") or []
                out.append({"names": names, "scores": scores})
            return out
    except Exception:
        pass
    return None


# ==============================================================================
# MAIN ANALYSIS FUNCTIONS
# ==============================================================================

def run_repeated_cv_analysis(X_base, y_base, X_csri, y_csri):
    """Run repeated 5-fold cross-validation with paired t-tests"""
    print("\n" + "="*60)
    print("APPROACH 1: Repeated 5-Fold CV (3 repetitions)")
    print("="*60)
    
    all_base_scores = {m: [] for m in ["accuracy", "precision", "recall", "f1"]}
    all_csri_scores = {m: [] for m in ["accuracy", "precision", "recall", "f1"]}
    
    for rep in range(3):
        skf_rep = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE+rep*100)
        splits_rep = list(skf_rep.split(X_base, y_base))
        
        scores_base_rep = scores_on_given_splits(X_base, y_base, splits_rep)
        scores_csri_rep = scores_on_given_splits(X_csri, y_csri, splits_rep)
        
        for metric in ["accuracy", "precision", "recall", "f1"]:
            all_base_scores[metric].extend(scores_base_rep[metric])
            all_csri_scores[metric].extend(scores_csri_rep[metric])
    
    print("\nRepeated CV Results (15 total folds):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        base_arr = np.array(all_base_scores[metric])
        csri_arr = np.array(all_csri_scores[metric])
        t_stat, p_val = paired_t_test_with_details(base_arr, csri_arr, metric)


def run_bootstrap_analysis(X_base, y_base, X_csri, y_csri):
    """Run bootstrap confidence interval analysis"""
    print("\n" + "="*60)
    print("APPROACH 2: Bootstrap Confidence Intervals")
    print("="*60)
    
    # Calculate scores using single 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_base, y_base))
    
    scores_base = scores_on_given_splits(X_base, y_base, splits)
    scores_csri = scores_on_given_splits(X_csri, y_csri, splits)
    
    print("\nBootstrap 95% CI for improvement (CSRI - Base):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        mean_diff, ci_low, ci_high = bootstrap_ci(scores_base[metric], scores_csri[metric], 
                                                   n_bootstrap=N_BOOTSTRAP)
        significant = "YES" if ci_low > 0 else "NO"
        print(f"{metric.upper():>9}: {mean_diff:.4f} [{ci_low:.4f}, {ci_high:.4f}] Significant: {significant}")


def run_ebm_explainability_analysis(csri_path: str):
    """Run EBM explainability analysis on CSRI model"""
    print("\n" + "="*60)
    print("EBM EXPLAINABILITY ANALYSIS (CSRI Model)")
    print("="*60)
    
    # Load and split data
    df = pd.read_csv(csri_path)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train EBM
    ebm = ExplainableBoostingClassifier(random_state=RANDOM_STATE)
    ebm.fit(X_train, y_train)
    
    # Global explanation
    global_df = extract_global_importance(ebm)
    global_df.to_csv("ebm_csri_global_importance.csv", index=False)
    
    # Visualisation
    plt.figure(figsize=(10, max(4, TOP_GLOBAL * 0.45)))
    top = global_df.head(TOP_GLOBAL).iloc[::-1]
    plt.barh(top["Feature (readable)"], top["Importance"])
    plt.xlabel("Importance (EBM)")
    plt.title("Top Feature Importances — EBM (CSRI)")
    plt.tight_layout()
    plt.savefig("ebm_csri_top_importances.png", dpi=150)
    plt.show()
    
    # Local explanations for highest-risk cases (if supported)
    proba = ebm.predict_proba(X_test)[:, 1]
    top_idx = np.argsort(-proba)[:TOP_AT_RISK_CASES]
    X_top = X_test.iloc[top_idx]
    y_top = y_test.iloc[top_idx].reset_index(drop=True)
    
    local_list = try_local_explanations(ebm, X_top)
    
    case_summaries = []
    if local_list is not None:
        for i in range(len(local_list)):
            contribs = local_list[i].get("scores") or []
            names = local_list[i].get("names") or []
            # Guard for numpy arrays
            if hasattr(contribs, "tolist"): contribs = contribs.tolist()
            if hasattr(names, "tolist"): names = names.tolist()
            pairs = sorted(zip(map(str, names), map(float, contribs)), key=lambda t: abs(t[1]), reverse=True)
            pairs = pairs[:TOP_LOCAL_FEATURES]
            summary_items = []
            for term, sc in pairs:
                feature_readable = prettify_name(term)
                direction = "increases risk" if sc > 0 else "reduces risk"
                summary_items.append(f"- {feature_readable}: {direction} (contribution {sc:+.3f})")
            case_summaries.append({
                "Case #": i + 1,
                "Test row id": int(X_top.index[i]),
                "Predicted risk": float(proba[top_idx[i]]),
                "Actual": int(y_top.iloc[i]),
                "Reasons": summary_items
            })
        
        # Save text + csv
        with open("ebm_csri_top_case_reasons.txt", "w") as f:
            f.write("EBM (CSRI) — Top At-Risk Cases and Reasons\n")
            f.write("=========================================\n\n")
            for case in case_summaries:
                f.write(f"Case {case['Case #']}: Row {case['Test row id']}, "
                       f"Risk={case['Predicted risk']:.2%}, Actual={case['Actual']}\n")
                for reason in case["Reasons"]:
                    f.write(f"  {reason}\n")
                f.write("\n")
        
        reasons_df = pd.DataFrame([
            {"Case": c["Case #"], "Row": c["Test row id"], 
             "Risk": c["Predicted risk"], "Actual": c["Actual"]}
            for c in case_summaries
        ])
        reasons_df.to_csv("ebm_csri_top_case_reasons.csv", index=False)
        
        print("\n--- Files written ---")
        print("1) ebm_csri_global_importance.csv")
        print("2) ebm_csri_top_importances.png")
        print("3) ebm_csri_top_case_reasons.txt")
        print("4) ebm_csri_top_case_reasons.csv")
    else:
        print("\n--- Files written ---")
        print("1) ebm_csri_global_importance.csv")
        print("2) ebm_csri_top_importances.png")
        print("Local explanations not available in this interpret version; global importances were saved.")
    
    print("\nTop global features:")
    print(global_df.head(TOP_GLOBAL)[["Feature (readable)", "Importance"]].to_string(index=False))


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run complete fairness testing analysis"""
    print("\n" + "="*80)
    print("FAIRNESS TESTING: BASE VS CSRI MODEL COMPARISON")
    print("="*80)
    
    # Load data
    X_base, y_base = load_xy(BASE_PATH)
    X_csri, y_csri = load_xy(CSRI_PATH)
    
    print(f"\nBase data shape: X={X_base.shape}, y={y_base.shape}")
    print(f"CSRI data shape: X={X_csri.shape}, y={y_csri.shape}")
    print(f"Base class distribution: {np.bincount(y_base)}")
    print(f"CSRI class distribution: {np.bincount(y_csri)}")
    
    # Run analyses
    run_repeated_cv_analysis(X_base, y_base, X_csri, y_csri)
    run_bootstrap_analysis(X_base, y_base, X_csri, y_csri)
    run_ebm_explainability_analysis(CSRI_PATH)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
