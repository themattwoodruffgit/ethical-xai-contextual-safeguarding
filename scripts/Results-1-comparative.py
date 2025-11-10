#!/usr/bin/env python3
"""
Comparative Results: 6 Algorithms × 3 Models + PyGOL Rule Generation
=====================================================================
This script evaluates six machine learning algorithms across three data models:
1. Baseline (base_exact_2031.csv)
2. All Data (full_exact_2031.csv)  
3. CSRI (csfri_exact_2031.csv)

Algorithms: LogisticRegression, RandomForest, DecisionTree, 
            GradientBoostedTree, ExplainableBoostingMachine, PyGOL

Outputs:
- Combined performance table
- Performance comparison visualization
- PyGOL learned rules for all models
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
import time
import glob
import PyGol as pygol

# ==============================================================================
# CONFIGURATION - MUST MATCH THESIS EXACTLY
# ==============================================================================

# CSV file paths (assumed to be in same directory as script)
FILES = [
    "base_exact_2031.csv",
    "full_exact_2031.csv",
    "csfri_exact_2031.csv",
]

# PyGOL parameters - EXACT configuration from thesis
PYGOL_PARAMS = {
    "max_literals": 3,
    "exact_literals": True,
    "key_size": 1,
    "min_pos": 12,
    "max_neg": 3,
    "default_div": 8,
    "verbose": True
}

# Background knowledge file
BK_PATH = "BK.pl"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def infer_model_label(p: str) -> str:
    """Infer model label from file path"""
    name = Path(p).name.lower()
    if "csri" in name or "csfri" in name:
        return "csri"
    if "full" in name:
        return "full"
    if "base" in name:
        return "base"
    return Path(p).stem.lower()


def prepare_xy(df: pd.DataFrame):
    """Prepare data for sklearn models (with one-hot encoding)"""
    df = df.assign(
        Current_Year=df['Current_NC_Year'].fillna(0) if 'Current_NC_Year' in df.columns else 0,
        Pupil_Premium_Indicator=df['Pupil_Premium_Indicator'].fillna("Unknown") if 'Pupil_Premium_Indicator' in df.columns else "Unknown",
        SEN_Status=df['SEN_Status'].fillna("Unknown") if 'SEN_Status' in df.columns else "Unknown",
    )
    for col in ['Gender', 'Pupil_Premium_Indicator', 'SEN_Status']:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col])
    features = df.columns.difference(['Is_CS_Incident', 'studentkey'])
    X = df[features]
    y = df['Is_CS_Incident']
    return X, y


def prepare_df_for_pygol(df: pd.DataFrame):
    """Prepare data for PyGOL (keeping categorical features intact)"""
    df_pygol = df.copy(deep=True)
    # Fill missing values but keep categorical columns as-is
    if 'Current_NC_Year' in df_pygol.columns:
        df_pygol['Current_Year'] = df_pygol['Current_NC_Year'].fillna(0)
    else:
        df_pygol['Current_Year'] = 0
    
    if 'Pupil_Premium_Indicator' in df_pygol.columns:
        df_pygol['Pupil_Premium_Indicator'] = df_pygol['Pupil_Premium_Indicator'].fillna("Unknown")
    else:
        df_pygol['Pupil_Premium_Indicator'] = "Unknown"
    
    if 'SEN_Status' in df_pygol.columns:
        df_pygol['SEN_Status'] = df_pygol['SEN_Status'].fillna("Unknown")
    else:
        df_pygol['SEN_Status'] = "Unknown"
    
    return df_pygol


def eval_ml_models(X_train, X_test, y_train, y_test):
    """Evaluate sklearn models with specificity calculation"""
    ml_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "GradientBoostedTree": GradientBoostingClassifier(random_state=42),
        "ExplainableBoostingMachine": ExplainableBoostingClassifier(random_state=42)
    }
    rows = []
    for name, clf in ml_models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        rows.append({
            "Algorithm": name,
            "Accuracy": rpt['accuracy'],
            "Precision": rpt['macro avg']['precision'],
            "Recall": rpt['macro avg']['recall'],
            "Specificity": specificity,
            "F1 score": rpt['macro avg']['f1-score'],
        })
    return rows


def eval_pygol_isolated(full_df: pd.DataFrame, tag: str, bk_path: str = "BK.pl",
                        max_literals=4, exact_literals=False, key_size=1,
                        min_pos=5, max_neg=60, default_div=6, verbose=False):
    """
    PyGOL evaluation with configurable parameters.
    
    Parameters:
    -----------
    max_literals : int - maximum number of conditions in a rule
    exact_literals : bool - whether to use exact matching
    key_size : int - size of feature combinations to consider
    min_pos : int - minimum positive examples a rule must cover
    max_neg : int - maximum negative examples a rule can cover
    default_div : int - number of discretisation bins
    verbose : bool - whether to print debug information
    """
    
    # Clean up old PyGOL files ONLY
    safe_patterns = [
        f"meta_{tag}_*.info",
        f"pos_{tag}_*.f",
        f"neg_{tag}_*.n",
        f"pos_full_{tag}.f",
        f"neg_full_{tag}.n",
        f"meta_{tag}.info"
    ]
    
    for pattern in safe_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                if verbose:
                    print(f"Removed old PyGOL file: {file}")
            except:
                pass
    
    # Prepare PyGOL-compatible dataframe
    df_pygol = prepare_df_for_pygol(full_df)
    
    # Create unique artifacts with timestamp
    timestamp = str(int(time.time() * 1000))
    meta = f"meta_{tag}_{timestamp}.info"
    pos_full = f"pos_{tag}_{timestamp}.f"
    neg_full = f"neg_{tag}_{timestamp}.n"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PyGOL Run: {tag}")
        print(f"Parameters: max_literals={max_literals}, exact_literals={exact_literals}")
        print(f"            key_size={key_size}, min_pos={min_pos}, max_neg={max_neg}")
        print(f"            default_div={default_div}")
        print(f"Files: {meta}, {pos_full}, {neg_full}")
        print(f"{'='*60}\n")
    
    feats = df_pygol.columns.difference(['Is_CS_Incident', 'studentkey'])
    
    # Step 1: Prepare logic rules with configurable discretisation
    pygol.prepare_logic_rules(
        df_pygol, feats,
        meta_information=meta,
        default_div=default_div,
        conditions={}
    )
    
    # Step 2: Prepare examples
    pygol.prepare_examples(
        df_pygol, 'Is_CS_Incident',
        positive_example=pos_full,
        negative_example=neg_full,
        meta_information=meta
    )
    
    # Step 3: Generate bottom clauses
    const = pygol.read_constants_meta_info(meta_information=meta)
    P, N = pygol.bottom_clause_generation(
        file=bk_path, 
        constant_set=const, 
        container="dict",
        positive_example=pos_full, 
        negative_example=neg_full
    )
    
    # Step 4: Split the data
    Train_P, Test_P, Train_N, Test_N = pygol.pygol_train_test_split(
        test_size=0.5, 
        positive_file_dictionary=P, 
        negative_file_dictionary=N
    )
    
    if verbose:
        print(f"Split: Train={len(Train_P)}P/{len(Train_N)}N, Test={len(Test_P)}P/{len(Test_N)}N")
    
    # Step 5: Learn model with CONFIGURABLE parameters
    model = pygol.pygol_learn(
        Train_P, Train_N,
        max_literals=max_literals,
        exact_literals=exact_literals,
        key_size=key_size,
        min_pos=min_pos,
        max_neg=max_neg
    )
    
    if verbose and hasattr(model, 'hypothesis'):
        print(f"Learned hypothesis length: {len(model.hypothesis)}")
        if len(model.hypothesis) < 500:
            print(f"Hypothesis: {model.hypothesis}")
    
    # Step 6: Evaluate
    m = pygol.evaluate_theory_prolog(
        model.hypothesis, 
        bk_path, 
        Test_P, 
        Test_N
    )
    
    # Clean up
    for f in [meta, pos_full, neg_full]:
        try:
            os.remove(f)
        except:
            pass
    
    return {
        "Algorithm": "PyGol",
        "Accuracy": m.accuracy,
        "Precision": m.precision,
        "Recall": m.sensitivity,
        "Specificity": m.specificity,
        "F1 score": m.fscore,
    }


def create_comparison_plot(df_table):
    """Create comprehensive comparison visualisation"""
    plot_df = df_table.copy()
    
    # Map model codes to display names
    model_name_map = {"base": "Baseline", "full": "All Data", "csri": "CSRI"}
    plot_df["Model"] = plot_df["Model"].str.lower().map(model_name_map)
    
    # Melt to long form
    long_df = plot_df.melt(
        id_vars=["Algorithm", "Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 score"],
        var_name="Metric",
        value_name="Value",
    )
    # Normalise metric naming
    long_df["Metric"] = long_df["Metric"].replace({"F1 score": "F1 Score"})
    
    # Ensure clean ordering
    metric_order  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    algo_order    = ["DecisionTree", "ExplainableBoostingMachine", "GradientBoostedTree",
                     "LogisticRegression", "PyGol", "RandomForest"]
    long_df["Metric"]    = pd.Categorical(long_df["Metric"], categories=metric_order, ordered=True)
    long_df["Algorithm"] = pd.Categorical(long_df["Algorithm"], categories=algo_order, ordered=True)
    
    # Pivot for plotting convenience
    pivot_df = (
        long_df.pivot_table(index=["Algorithm","Model"], columns="Metric", values="Value")
                .reset_index()
    )
    
    algos  = [a for a in algo_order if a in pivot_df["Algorithm"].unique()]
    models = ["Baseline", "All Data", "CSRI"]
    metric_nums   = {m: str(i + 1) for i, m in enumerate(metric_order)}
    model_letters = dict(zip(models, list("ABC")))
    
    # Plotting setup
    colors  = {"Baseline": "#1f77b4", "All Data": "#ff7f0e", "CSRI": "#2ca02c"}
    hatches = {"Accuracy": "", "Precision": "///", "Recall": "\\\\", "F1 Score": "xx"}
    
    bar_w        = 0.04
    cluster_w    = len(models) * len(metric_order) * bar_w
    cluster_gap  = bar_w * 2
    
    fig, ax = plt.subplots(figsize=(16, 8))
    xticks, xtick_labels = [], []
    row_h = 0.045
    
    # Draw bars & labels
    for a_idx, algo in enumerate(algos):
        cluster_c = a_idx * (cluster_w + cluster_gap)
    
        for m_idx, model in enumerate(models):
            for t_idx, metric in enumerate(metric_order):
                # Value lookup
                sel = pivot_df[(pivot_df["Algorithm"] == algo) & (pivot_df["Model"] == model)]
                if sel.empty: 
                    continue
                y = float(sel[metric].values[0])
    
                x = cluster_c - cluster_w/2 + bar_w/2 + (m_idx*len(metric_order) + t_idx) * bar_w
                ax.bar(
                    x, y,
                    width=bar_w, color=colors[model], hatch=hatches[metric],
                    edgecolor="black", linewidth=0.6,
                )
                xticks.append(x)
                xtick_labels.append(metric_nums[metric])
    
        # Model letters under each group
        for m_idx, model in enumerate(models):
            model_c = cluster_c - cluster_w/2 + bar_w/2 + (m_idx*len(metric_order) + len(metric_order)/2 - 0.5) * bar_w
            ax.text(
                model_c, -row_h, model_letters[model],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold",
            )
    
        # Full algorithm name label below letters
        ax.text(
            cluster_c, -2*row_h, algo,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9,
        )
    
    # Axes, grid, legends
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    
    ax.set_xlabel("Metric  •  Model  •  Algorithm", fontweight="bold", labelpad=20)
    ax.xaxis.set_label_coords(0.5, -3*row_h - 0.02)
    
    ax.set_ylabel("Performance Score", fontweight="bold")
    ax.set_title("Model & Algorithm Performance Comparison", fontweight="bold", pad=20)
    
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(0.30, 0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    model_handles = [Patch(facecolor=colors[m], edgecolor="black", label=f"{model_letters[m]} {m}") for m in models]
    metric_handles = [Patch(facecolor="white", edgecolor="black", hatch=hatches[m], label=f"{metric_nums[m]} {m}") for m in metric_order]
    
    lg1 = ax.legend(handles=model_handles, title="Models", loc="upper right")
    ax.add_artist(lg1)
    ax.legend(handles=metric_handles, title="Metrics", loc="upper right", bbox_to_anchor=(1, 0.74))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, bottom=0.35)
    plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("model_performance_comparison.pdf", bbox_inches="tight")
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run comparative analysis"""
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS: 6 ALGORITHMS × 3 MODELS")
    print("="*80)
    
    print(f"\n{'='*60}")
    print("RUNNING WITH PYGOL PARAMETERS:")
    for k, v in PYGOL_PARAMS.items():
        if k != "verbose":
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")
    
    all_rows = []
    
    for fpath in FILES:
        model_label = infer_model_label(fpath)
        df = pd.read_csv(fpath)
        
        print(f"\nProcessing {model_label} model...")
        
        # sklearn pipeline
        X, y = prepare_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Evaluate sklearn models
        for r in eval_ml_models(X_train, X_test, y_train, y_test):
            r["Model"] = model_label
            all_rows.append(r)
        
        # PyGOL evaluation with configurable parameters
        pyg_row = eval_pygol_isolated(df, model_label, bk_path=BK_PATH, **PYGOL_PARAMS)
        pyg_row["Model"] = model_label
        all_rows.append(pyg_row)
        
        print(f"PyGOL {model_label}: Recall={pyg_row['Recall']:.3f}, Specificity={pyg_row['Specificity']:.3f}")
    
    # Final table
    df_table = pd.DataFrame(all_rows, 
                            columns=["Algorithm", "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1 score"])
    
    # Sort
    model_order = {"base": 0, "full": 1, "csri": 2}
    algo_order = {
        "LogisticRegression": 0,
        "RandomForest": 1,
        "DecisionTree": 2,
        "GradientBoostedTree": 3,
        "ExplainableBoostingMachine": 4,
        "PyGol": 5,
    }
    df_table["_a"] = df_table["Algorithm"].map(algo_order).fillna(999)
    df_table["_m"] = df_table["Model"].str.lower().map(model_order).fillna(999)
    df_table = df_table.sort_values(by=["_a","_m","Algorithm","Model"]).drop(columns=["_a","_m"]).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("COMBINED MODEL PERFORMANCE (base/full/csri)")
    print("="*60)
    print(df_table.to_string())
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"PyGOL Average Recall: {df_table[df_table['Algorithm']=='PyGol']['Recall'].mean():.3f}")
    print(f"PyGOL Average Specificity: {df_table[df_table['Algorithm']=='PyGol']['Specificity'].mean():.3f}")
    print(f"Other Models Average Recall: {df_table[df_table['Algorithm']!='PyGol']['Recall'].mean():.3f}")
    print(f"Other Models Average Specificity: {df_table[df_table['Algorithm']!='PyGol']['Specificity'].mean():.3f}")
    
    # Calculate false positive rates for operational impact
    for model_label in ["base", "full", "csri"]:
        pygol_row = df_table[(df_table['Algorithm']=='PyGol') & (df_table['Model']==model_label)].iloc[0]
        best_other = df_table[(df_table['Algorithm']!='PyGol') & (df_table['Model']==model_label)].nlargest(1, 'F1 score').iloc[0]
        
        print(f"\n{model_label.upper()} Model Comparison:")
        print(f"  PyGOL - Recall: {pygol_row['Recall']:.3f}, Specificity: {pygol_row['Specificity']:.3f}")
        print(f"  {best_other['Algorithm']} - Recall: {best_other['Recall']:.3f}, Specificity: {best_other['Specificity']:.3f}")
        
        # Estimate operational impact
        false_positive_rate_pygol = 1 - pygol_row['Specificity']
        false_positive_rate_other = 1 - best_other['Specificity']
        print(f"  Per 1000 students, PyGOL flags ~{false_positive_rate_pygol*1000:.0f} false positives")
        print(f"  Per 1000 students, {best_other['Algorithm']} flags ~{false_positive_rate_other*1000:.0f} false positives")
    
    # Create visualisation
    create_comparison_plot(df_table)
    
    # Save results
    df_table.to_csv("combined_results.csv", index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files saved:")
    print("  - combined_results.csv")
    print("  - model_performance_comparison.png")
    print("  - model_performance_comparison.pdf")


if __name__ == "__main__":
    main()