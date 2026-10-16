"""Evaluate classification thresholds for bad-credit-risk detection."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'people_liable', 'telephone',
    'foreign_worker', 'risk_label'
]
CATEGORICAL = [
    'checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
    'personal_status', 'other_debtors', 'property', 'other_installment_plans',
    'housing', 'job', 'telephone', 'foreign_worker'
]
NUMERIC = [column for column in COLUMNS[:-1] if column not in CATEGORICAL]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=Path(__file__).with_name('german.data'),
        help='Path to UCI german.data',
    )
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f'Dataset not found: {data_path}')

    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=COLUMNS)
    if not set(df['risk_label'].unique()).issubset({1, 2}):
        raise ValueError('Expected original UCI risk labels 1 (good) and 2 (bad).')
    X = df.drop(columns='risk_label')
    y = df['risk_label'].eq(2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC),
        ('cat', categorical_pipeline, CATEGORICAL),
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        rows.append({
            'Threshold': threshold,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3),
            'Missed Bad Risks': int(matrix[1, 0]),
            'False Alarms': int(matrix[0, 1]),
        })

    results = pd.DataFrame(rows)
    print('Precision and recall treat bad risk (original label 2) as positive.')
    print(results.to_string(index=False))

    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    print(f'\nSaved threshold results to {output_dir / "threshold_analysis.csv"}')


if __name__ == '__main__':
    main()