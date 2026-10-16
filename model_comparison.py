"""Compare Logistic Regression and Random Forest on the German Credit data."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
import sklearn.model_selection
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


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocess = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC),
        ('cat', categorical_pipeline, CATEGORICAL),
    ])
    return preprocess


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
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
        ),
    }
    rows = []
    for name, classifier in models.items():
        pipeline = Pipeline([
            ('preprocessor', make_preprocessor()),
            ('classifier', classifier),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, predictions), 3),
            'Precision': round(precision_score(y_test, predictions, zero_division=0), 3),
            'Recall': round(recall_score(y_test, predictions, zero_division=0), 3),
            'ROC-AUC': round(roc_auc_score(y_test, probabilities), 3),
        })
        print(f'\n{name}')
        print('Confusion Matrix (rows=true, columns=predicted; order=good, bad):')
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results = pd.DataFrame(rows)
    print('\nMODEL COMPARISON')
    print('Precision and recall treat bad risk (original label 2) as positive.')
    print(results.to_string(index=False))

    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f'\nSaved comparison results to {output_dir / "model_comparison.csv"}')


if __name__ == '__main__':
    main()