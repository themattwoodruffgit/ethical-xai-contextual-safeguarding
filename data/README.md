# Data Directory

This directory contains the CSV files required for analysis. **These three specific CSV files are included in the repository to enable reproduction of thesis results.**

⚠️ **Important**: These files contain anonymized data for research purposes. The .gitignore is configured to prevent accidentally committing any other CSV files to the repository.

## Included Files

The analysis scripts require the following CSV files (all included):

### 1. `base_exact_2031.csv`
**Baseline demographic model** containing only basic demographic features.

**Required columns**:
- `FSM_EverFSM6` - Pupil premium eligibility (protected attribute)
- `CurrentNCYear` - Current national curriculum year
- `Gender` - Student gender
- `IDACI_Score` - Deprivation index score
- `SEN` - Special Educational Needs status
- `EAL` - English as Additional Language
- `Ethnicity` - Ethnic background
- `safeguarding_any` - Target variable (1 = safeguarding concern, 0 = no concern)

### 2. `full_exact_2031.csv`
**Full feature set** including demographics, attendance, and behaviour.

**Required columns**:
- All columns from `base_exact_2031.csv`, plus:
- `attendance_score` - Attendance metric
- `behaviour_points` - Behaviour incident count
- `exclusions` - Exclusion count
- Additional contextual features as available

### 3. `csfri_exact_2031.csv`
**Contextual Safeguarding Risk Indicators (CSRI)** - the main model evaluated in the thesis.

**Required columns**:
- All columns from `full_exact_2031.csv`, plus:
- `peer_relationships` - Peer relationship indicators
- `extra_familial_risks` - Extra-familial harm indicators
- `neighbourhood_risks` - Community-level risk factors
- `online_safety` - Online safety concerns
- Other contextual safeguarding indicators

### 4. `BK.pl`
**PyGOL background knowledge file** - Defines logical predicates and modes for inductive logic programming.

This file is included in the repository as it contains no sensitive information, only logical rule definitions.

## Data Format Requirements

### General Requirements
- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)
- **Missing values**: Use `NaN` or leave empty
- **Target variable**: Binary (0 or 1)
- **Protected attribute**: Binary (0 or 1) for pupil premium status

### Data Types
- **Categorical features**: Should be one-hot encoded or label encoded
- **Numerical features**: Continuous or discrete numeric values
- **Binary features**: 0 or 1

### Example Data Structure

```csv
FSM_EverFSM6,CurrentNCYear,Gender,IDACI_Score,SEN,EAL,Ethnicity,safeguarding_any
1,9,1,0.45,0,0,3,0
0,10,0,0.23,1,0,1,1
1,8,1,0.67,0,1,2,1
...
```

## Ethical Considerations

### Data Protection
The three CSV files in this directory are **included in the repository** to enable thesis reproduction. These files contain anonymized data that has been processed according to:
- GDPR compliance requirements
- Data protection regulations
- Safeguarding children's privacy
- Ethical research practice

⚠️ **Note**: The `.gitignore` is configured to prevent accidentally committing any OTHER CSV files to the repository. Only the three specific 2031 files are allowed.

### Required Approvals
Before using these scripts with real data, you must have:
- ✅ Ethical approval from relevant research ethics committee
- ✅ Data protection impact assessment (DPIA)
- ✅ Legal basis for processing under GDPR
- ✅ Appropriate data sharing agreements
- ✅ Secure data storage arrangements

### Anonymisation
Data should be:
- Free of personally identifiable information (PII)
- Using pseudonymised student IDs if needed for tracking
- Aggregated where possible to reduce re-identification risk
- Stored securely with appropriate access controls

## Preparing Your Data

### Step 1: Data Extraction
Extract data from your source system ensuring:
- All required columns are present
- Data types are consistent
- Missing values are appropriately handled

### Step 2: Feature Engineering
Create derived features as needed:
- Attendance scores
- Behaviour point aggregations
- Contextual safeguarding indicators

### Step 3: Anonymisation
Remove or pseudonymise:
- Student names
- Addresses
- Date of birth (convert to age or year group)
- Any other PII

### Step 4: Validation
Check your data:
```bash
# Row count
wc -l data/base_exact_2031.csv

# Column names
head -1 data/base_exact_2031.csv

# Basic statistics
python -c "import pandas as pd; print(pd.read_csv('data/base_exact_2031.csv').describe())"
```

## File Sizes

Typical file sizes (for reference):
- `base_exact_2031.csv`: ~500KB - 2MB
- `full_exact_2031.csv`: ~1MB - 5MB
- `csfri_exact_2031.csv`: ~2MB - 10MB
- `BK.pl`: ~700KB (included in repository)

Actual sizes will vary based on your dataset size and number of features.

## Troubleshooting

### Error: "File not found"
Ensure CSV files are in the `data/` directory and named exactly as expected.

### Error: "KeyError: column_name"
Check that all required columns are present in your CSV files.

### Error: "Memory error"
For very large datasets:
- Consider sampling for initial testing
- Use chunked reading: `pd.read_csv(..., chunksize=10000)`
- Ensure sufficient RAM available

### Error: "Encoding error"
Ensure files are UTF-8 encoded:
```bash
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv
```

## Data Quality Checks

Before running analyses, verify:
- [ ] No missing values in target variable (`safeguarding_any`)
- [ ] Protected attribute (`FSM_EverFSM6`) has both classes represented
- [ ] No duplicated rows
- [ ] Appropriate class balance (not too imbalanced)
- [ ] Feature distributions look sensible
- [ ] No data leakage (future information in predictors)

## Further Information

For questions about data format or preparation:
1. Review thesis methodology chapter
2. Check example data format in `outputs/` directory
3. Contact repository maintainer

---

**Remember**: This is sensitive safeguarding data. Handle with appropriate care and comply with all relevant data protection regulations.
