# Credit Risk Analysis & Classification

### Machine Learning | Python | Scikit-learn | Pandas | Data Analytics

## Project Overview

Developed a machine learning pipeline to analyze borrower credit characteristics and classify applicants into good-risk and bad-risk categories.

The project demonstrates data preprocessing, exploratory data analysis (EDA), statistical modelling, model evaluation, and classification threshold analysis to support data-driven credit risk assessment.

**Note:** The dataset provides historical credit-risk labels rather than verified loan-default events. This project is a credit-risk classification exercise, not a validated loan-default probability model.

## Dataset

**Source:** [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

- Total records: 1,000
- Input features: 20
- Good credit risk: 700 (70%)
- Bad credit risk: 300 (30%)
- Missing values: 0

The dataset contains borrower and credit characteristics including credit amount, loan duration, credit history, employment information, and savings status.

## Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Algorithms:** Logistic Regression, Random Forest

## Project Workflow

### 1. Data Preprocessing

- Loaded and inspected the credit dataset.
- Identified numerical and categorical features.
- Applied feature scaling and one-hot encoding.
- Performed stratified 80/20 train-test splitting.
- Used training-only preprocessing to prevent data leakage.

### 2. Exploratory Data Analysis

Analyzed credit-risk distributions and relationships between borrower characteristics and risk categories.

Generated visualizations covering:

- Distribution of good and bad credit risks.
- Credit amount across risk categories.
- Loan duration across risk categories.

### 3. Machine Learning Models

Implemented and compared two classification algorithms:

- Logistic Regression
- Random Forest Classifier

Both models were evaluated on the same held-out test set.

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 78.0% | 75.0% |
| Precision (Bad Risk) | 66.7% | 65.6% |
| Recall (Bad Risk) | 53.3% | 35.0% |
| ROC-AUC | 0.804 | 0.796 |

**Key Finding:** Logistic Regression outperformed Random Forest across the reported metrics on this test split.

## Classification Threshold Analysis

Analyzed the effect of changing the classification threshold on bad-risk detection.

| Threshold | Precision | Recall | F1-Score | Missed Bad-Risk Cases |
|---|---:|---:|---:|---:|
| 0.30 | 56.3% | 81.7% | 0.667 | 11 |
| 0.40 | 63.2% | 71.7% | 0.672 | 17 |
| 0.50 | 66.7% | 53.3% | 0.593 | 28 |
| 0.60 | 70.6% | 40.0% | 0.511 | 36 |

**Key Finding:** Reducing the threshold from 0.50 to 0.40 improved bad-risk recall from 53.3% to 71.7%, reducing missed bad-risk cases from 28 to 17 while increasing false positives from 16 to 25.

The 0.40 threshold achieved the highest F1-score among the evaluated thresholds. These comparisons are exploratory and do not establish an optimal production threshold.

## How to Run

**1. Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
cd Credit-Risk-Analysis
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `german.data` from the UCI dataset link above and place it in the project directory.

**4. Run the analysis**

```bash
python eda.py
python train.py --data german.data
python model_comparison.py
python threshold_analysis.py
```

Generated evaluation results and visualizations are saved in the `outputs/` directory.

## Key Learnings

- Applied machine learning to a financial risk classification problem.
- Developed an understanding of feature preprocessing and classification algorithms.
- Evaluated model performance using multiple metrics rather than accuracy alone.
- Investigated the trade-off between identifying risky borrowers and incorrectly flagging good-risk borrowers.
- Practiced reproducible experimentation and technical documentation.

## Limitations and Future Improvements

- The dataset is small and historical and does not represent modern corporate or institutional lending portfolios.
- Model performance was evaluated using a single train-test split.
- Threshold analysis used the held-out test set for exploratory comparison; future work should select thresholds using validation data and reserve a separate final test set.
- Future improvements include cross-validation, probability calibration, model interpretability, fairness assessment, and temporal model monitoring.
- The model is intended for educational analysis and should not be used for real lending decisions.
