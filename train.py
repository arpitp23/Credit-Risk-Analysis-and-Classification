"""Credit risk classification using the UCI Statlog German Credit dataset.
Run: python train.py --data german.data
Download german.data from https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
"""
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
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
NUMERIC = [c for c in COLUMNS[:-1] if c not in CATEGORICAL]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='german.data', help='Path to UCI german.data')
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f'Dataset not found: {path}. Download german.data from the UCI dataset page.')
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    if not set(df['risk_label'].unique()).issubset({1, 2}):
        raise ValueError('Expected original UCI risk labels 1 (good) and 2 (bad).')
    # UCI: 1 = good credit risk; 2 = bad credit risk.
    y = (df['risk_label'] == 2).astype(int)
    X = df.drop(columns='risk_label')
    print('Rows:', len(df), 'Bad credit risk rate:', f'{y.mean():.1%}')
    print('Missing values:', int(X.isna().sum().sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
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
    model = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    print('ROC-AUC:', round(roc_auc_score(y_test, probabilities), 3))
    print('Precision:', round(precision_score(y_test, predictions), 3))
    print('Recall:', round(recall_score(y_test, predictions), 3))
    print('Confusion Matrix (rows=true, columns=predicted; order=good, bad):')
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(classification_report(y_test, predictions, target_names=['Good risk', 'Bad risk']))
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Good', 'Bad'])
    plt.title('Credit risk confusion matrix')
    plt.tight_layout()
    plt.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close()
    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title('Credit risk ROC curve')
    plt.tight_layout()
    plt.savefig(out / 'roc_curve.png', dpi=150)
    plt.close()
    joblib.dump(model, out / 'credit_risk_model.joblib')
    print('Saved charts and model in outputs/')


if __name__ == '__main__':
    main()
