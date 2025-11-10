# Ethical XAI-based Contextual Safeguarding

Implementation code for PhD thesis: **Ethical Explainable AI-based Contextual Safeguarding**

## Overview

This repository contains the complete implementation of a novel approach to safeguarding in educational contexts using explainable AI (XAI). The research addresses critical fairness concerns in predictive safeguarding systems while maintaining high predictive performance and providing transparent, interpretable explanations.

## Key Contributions

1. **Comparative Analysis**: Evaluation of 6 machine learning algorithms across 3 different data models for safeguarding prediction
2. **Fairness Testing**: Statistical analysis of bias with respect to pupil premium status, using Explainable Boosting Machines (EBM)
3. **Hybrid XAI Model**: Novel confidence-based routing system combining EBM and PyGOL (Inductive Logic Programming) to address fairness concerns while maintaining predictive performance

## Repository Structure

```
ethical-xai-contextual-safeguarding/
├── scripts/                        # Main analysis scripts
│   ├── Results-1-comparative.py   # Comparative algorithm analysis
│   ├── Results-2-fairness-ebm.py  # Fairness testing with EBM
│   ├── Results-3-hybrid.py        # Hybrid EBM-PyGOL model
│   └── README.md                  # Detailed script documentation
├── data/                          # Data directory (CSV files not committed)
│   ├── BK.pl                      # PyGOL background knowledge
│   └── README.md                  # Data requirements
├── outputs/                       # Analysis outputs
│   ├── comparative/               # Results from script 1
│   ├── fairness/                  # Results from script 2
│   ├── hybrid/                    # Results from script 3
│   └── README.md                  # Output documentation
├── docs/                          # Additional documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/themattwoodruffgit/ethical-xai-contextual-safeguarding.git
cd ethical-xai-contextual-safeguarding
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install git+https://github.com/logic-and-learning-lab/PyGol.git
```

3. **Run the analyses** (data files are already included):
   - The three required CSV files are included in the `data/` directory
   - See `data/README.md` for data format documentation

4. **Execute the scripts**:
```bash
cd scripts
python3 Results-1-comparative.py
python3 Results-2-fairness-ebm.py
python3 Results-3-hybrid.py
```

## Included Data Files

The scripts require three CSV files (**included in repository**):

- `data/base_exact_2031.csv` - Baseline demographic model (60KB)
- `data/full_exact_2031.csv` - Full feature set model (68KB)
- `data/csfri_exact_2031.csv` - Contextual Safeguarding Risk Indicators (CSRI) model (96KB)

These files contain anonymized data for thesis reproduction. See `data/README.md` for detailed column descriptions and ethical considerations.

## Key Findings

- **Comparative Analysis**: PyGOL achieved 74.7% recall with 47.1% specificity on CSRI model, providing rule-based explanations
- **Fairness Testing**: CSRI model showed statistically significant improvements in fairness metrics compared to baseline
- **Hybrid Model**: Novel EBM-PyGOL routing achieved balanced performance across demographic groups while maintaining interpretability

## Methodology

### Algorithms Evaluated
- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosted Tree
- Explainable Boosting Machine (EBM)
- PyGOL (Inductive Logic Programming)

### Fairness Metrics
- False Positive Rate (FPR) gap
- True Positive Rate (TPR) gap
- Statistical parity difference
- Equalised odds

### Explainability Approaches
- **Global**: Feature importance rankings, learned logical rules
- **Local**: Instance-level explanations for individual predictions

## Ethical Considerations

This research involves sensitive data related to child safeguarding. The data files included in this repository:
- Have been anonymized and processed according to ethical approvals
- **ARE included** (3 CSV files) to enable thesis reproduction
- Were handled according to GDPR and data protection regulations
- Have undergone appropriate ethical review

**Important**: The included data files are for academic research and thesis examination purposes. If you wish to adapt these methods for your own safeguarding context, you must obtain appropriate ethical approval and data protection clearance.

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{woodruff2025ethical,
  title={Ethical Explainable AI-based Contextual Safeguarding},
  author={Woodruff, Matthew},
  year={2025},
  school={University of Surrey},
  note={PhD Thesis}
}
```

## Documentation

- **Scripts**: See `scripts/README.md` for detailed usage instructions
- **Data**: See `data/README.md` for data format requirements
- **Outputs**: See `outputs/README.md` for output file descriptions

## Technical Details

- **Random Seed**: 42 (for reproducibility)
- **Train/Test Split**: 50-50 for most analyses
- **Cross-validation**: 5-fold repeated 3 times (15 folds total)
- **Bootstrap Iterations**: 1000 for confidence intervals
- **PyGOL Configuration**: max_literals=3, min_pos=12, max_neg=3

## Contact

**Author**: Matthew Woodruff
**Institution**: The University of Surrey
**Email**: themattwoodruff@gmail.com

## Acknowledgements

This research builds on the PyGOL framework.

## Status

This repository contains the implementation code for a PhD thesis currently under examination. The code is provided for reproducibility and transparency purposes.

---

**Last Updated**: November 2025
