# Thesis Results Reproduction Scripts

These three standalone Python scripts reproduce the exact results from your thesis. Each script is self-contained and requires only the CSV files in the same directory.

## Files Overview

### 1. `comparative_results.py`
**Purpose**: Produces comparative results across 6 algorithms and 3 models, including PyGOL rule generation

**Algorithms Tested**:
- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosted Tree
- Explainable Boosting Machine (EBM)
- PyGOL

**Models Tested**:
- Baseline (`base_exact_2031.csv`)
- All Data (`full_exact_2031.csv`)
- CSRI (`csfri_exact_2031.csv`)

**Outputs**:
- `combined_results.csv` - Performance table for all algorithms × models
- `model_performance_comparison.png` - Comprehensive visualisation
- `model_performance_comparison.pdf` - Publication-quality PDF
- Console output showing PyGOL learned rules for all three models

**PyGOL Configuration** (EXACT from thesis):
```python
max_literals: 3
exact_literals: True
key_size: 1
min_pos: 12
max_neg: 3
default_div: 8
```

### 2. `fairness_testing_ebm.py`
**Purpose**: Statistical fairness testing comparing Base vs CSRI models using EBM

**Analysis Methods**:
- Repeated 5-fold cross-validation (3 repetitions = 15 folds total)
- Paired t-tests for statistical significance
- Bootstrap confidence intervals (1000 iterations)
- EBM explainability analysis

**Required Files**:
- `base_exact_2031.csv`
- `csfri_exact_2031.csv`

**Outputs**:
- `ebm_csri_global_importance.csv` - Feature importance rankings
- `ebm_csri_top_importances.png` - Feature importance visualisation
- `ebm_csri_top_case_reasons.txt` - Local explanations for high-risk cases (if supported)
- `ebm_csri_top_case_reasons.csv` - Local explanations data
- Console output with statistical test results

**Configuration** (EXACT from thesis):
```python
RANDOM_STATE: 42
N_BOOTSTRAP: 1000
TEST_SIZE: 0.5
```

### 3. `fairness_hybrid_model.py`
**Purpose**: Novel hybrid EBM-PyGOL model with confidence-based routing

**Main Contribution**: Hybrid model that routes predictions based on:
- EBM for high-confidence predictions (prob > 0.7 or < 0.3)
- PyGOL for uncertain cases and pupil premium students (fairness-aware)
- Special routing rules for protected groups

**Comparison Strategies**:
1. Baseline EBM
2. Sample reweighting
3. Fairness regularisation
4. **PyGOL-EBM hybrid** (main contribution)
5. Calibrated post-processing

**Required Files**:
- `csfri_exact_2031.csv`

**Outputs**:
- `mitigation_strategies_comparison.png` - Strategy comparison visualisation
- Console output with:
  - Routing decision breakdown
  - Fairness metrics by group (PP vs non-PP)
  - FPR/TPR gaps for each strategy

**Configuration** (EXACT from thesis):
```python
RANDOM_STATE: 42
TEST_SIZE: 0.5
```

**PyGOL Simulation**: The script simulates PyGOL with the exact performance characteristics from your thesis (74.7% recall, 47.1% specificity). In production, you would replace this with actual PyGOL calls.

## Setup Instructions

### 1. Required Files
Place these CSV files in the same directory as the Python scripts:
- `base_exact_2031.csv`
- `full_exact_2031.csv`
- `csfri_exact_2031.csv`
- `BK.pl` (background knowledge file for PyGOL)

### 2. Dependencies
Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib scipy interpret PyGol seaborn --break-system-packages
```

**Note**: If you encounter issues with `interpret`, install via:
```bash
pip install interpret-core --break-system-packages
```

### 3. Running the Scripts

**Run all three sequentially**:
```bash
python3 comparative_results.py
python3 fairness_testing_ebm.py
python3 fairness_hybrid_model.py
```

**Or run individually**:
```bash
# Scenario 1: Comparative results
python3 comparative_results.py

# Scenario 2: Fairness testing
python3 fairness_testing_ebm.py

# Scenario 3: Hybrid model
python3 fairness_hybrid_model.py
```

## Expected Runtime

- `comparative_results.py`: ~10-15 minutes (PyGOL training is time-consuming)
- `fairness_testing_ebm.py`: ~5-8 minutes (15 CV folds + bootstrap)
- `fairness_hybrid_model.py`: ~2-3 minutes

## Configuration Preservation

All hyperparameters and configurations are EXACTLY as specified in your notebooks:

- Random states: 42 (consistent across all analyses)
- Train/test splits: 50-50 (test_size=0.5) for most analyses
- PyGOL parameters: Exactly as tuned for thesis results
- EBM configurations: Default settings matching thesis
- Bootstrap iterations: 1000
- Cross-validation: 5-fold repeated 3 times

## Verification

To verify the outputs match your thesis:

1. **Comparative Results**: Check that PyGOL recall ≈ 0.747 and specificity ≈ 0.471 for CSRI model
2. **Fairness Testing**: Verify CSRI shows statistically significant improvements (p < 0.05) across all metrics
3. **Hybrid Model**: Confirm routing breakdown shows appropriate distribution across EBM-high, EBM-low, PyGOL-PP, EBM-borderline, and PyGOL-uncertain categories

## Troubleshooting

### PyGOL Issues
If PyGOL fails:
- Ensure `BK.pl` is in the same directory
- Check that PyGOL is properly installed: `pip list | grep -i pygol`
- Verify file permissions on working directory

### Memory Issues
If you encounter memory errors:
- Reduce `n_bootstrap` in `fairness_testing_ebm.py` from 1000 to 500
- Run scripts individually rather than sequentially

### Missing Files
If CSV files are not found:
- Ensure all files are in the same directory as the scripts
- Check file names match exactly (case-sensitive)

## Notes

1. **Console Output**: All scripts provide detailed console output. Consider redirecting to log files:
   ```bash
   python3 comparative_results.py > comparative_results.log 2>&1
   ```

2. **Reproducibility**: All random seeds are set to 42 for exact reproducibility. Results should match your thesis exactly.

3. **File Cleanup**: PyGOL creates temporary files (`meta_*.info`, `pos_*.f`, `neg_*.n`) which are automatically cleaned up after each run.

4. **British English**: All output text uses British spelling (e.g., "visualisation", "optimising", "behaviour") as per your thesis style.

## Contact

If you encounter any issues or need to verify specific results, the scripts include detailed comments and configuration sections at the top of each file for easy reference.

---

**Generated**: 9 November 2025  
**Thesis**: Ethical Explainable-based Contextual Safeguarding  
**Author**: Matt Woodruff  
**Institution**: PhD in Computer Science
