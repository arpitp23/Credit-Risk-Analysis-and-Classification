"""Explore the UCI Statlog German Credit dataset."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

COLUMNS = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'existing_credits', 'job', 'people_liable', 'telephone',
    'foreign_worker', 'risk_label'
]


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
    df['risk'] = df['risk_label'].map({1: 'Good risk', 2: 'Bad risk'})

    print(f'Rows: {len(df)}; features: {len(COLUMNS) - 1}')
    print('Duplicate rows:', int(df.duplicated().sum()))
    print('Missing values:', int(df.isna().sum().sum()))
    print('\nRisk label counts:')
    print(df['risk'].value_counts().to_string())
    print('\nNumeric feature summary:')
    print(df[['duration', 'credit_amount', 'age']].describe().round(1).to_string())

    duration_labels = ['<=12', '13-24', '25-36', '37-48', '49-60', '>60']
    df['duration_band'] = pd.cut(
        df['duration'],
        bins=[0, 12, 24, 36, 48, 60, float('inf')],
        labels=duration_labels,
    )
    duration_summary = (
        df.assign(is_bad_risk=df['risk_label'].eq(2))
        .groupby('duration_band', observed=False)
        .agg(
            applicants=('risk_label', 'size'),
            bad_risk_share=('is_bad_risk', 'mean'),
        )
        .reset_index()
    )
    duration_summary['bad_risk_share'] *= 100
    print('\nRisk label by loan duration:')
    print(
        duration_summary.rename(columns={'bad_risk_share': 'bad_risk_share_%'})
        .round({'bad_risk_share_%': 1})
        .to_string(index=False)
    )

    output_dir = Path(__file__).parent / 'outputs' / 'eda'
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='risk', order=['Good risk', 'Bad risk'], ax=ax)
    ax.set(title='Credit risk label counts', xlabel='', ylabel='Applicants')
    fig.tight_layout()
    fig.savefig(output_dir / 'risk_label_counts.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for column, ax in zip(['duration', 'credit_amount', 'age'], axes):
        sns.boxplot(data=df, x='risk', y=column, order=['Good risk', 'Bad risk'], ax=ax)
        ax.set_title(column.replace('_', ' ').title())
        ax.set_xlabel('')
    fig.suptitle('Numeric features by credit risk label')
    fig.tight_layout()
    fig.savefig(output_dir / 'numeric_features_by_risk.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(duration_summary))
    ax.bar(positions, duration_summary['applicants'], color='#2a9d8f')
    ax.set_xticks(list(positions), duration_summary['duration_band'].astype(str))
    ax.set_xlabel('Loan duration (months)')
    ax.set_ylabel('Applicants')
    ax.set_title('Observed bad-risk label share by loan duration')
    rate_ax = ax.twinx()
    rate_ax.plot(
        positions,
        duration_summary['bad_risk_share'] / 100,
        color='#e76f51',
        marker='o',
    )
    rate_ax.set_ylabel('Bad-risk label share')
    rate_ax.set_ylim(0, 1)
    rate_ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(output_dir / 'risk_by_loan_duration.png', dpi=150)
    plt.close(fig)

    category_columns = ['checking_status', 'purpose', 'employment']
    fig, axes = plt.subplots(1, len(category_columns), figsize=(18, 6))
    for column, ax in zip(category_columns, axes):
        bad_risk_rate = (
            df.assign(is_bad_risk=df['risk_label'].eq(2))
            .groupby(column, observed=True)['is_bad_risk']
            .mean()
            .sort_values()
        )
        bad_risk_rate.plot.barh(ax=ax)
        ax.set_title(column.replace('_', ' ').title())
        ax.set_xlabel('Bad risk share')
        ax.set_ylabel('')
        ax.set_xlim(0, 1)
    fig.suptitle('Bad risk share by category')
    fig.tight_layout()
    fig.savefig(output_dir / 'bad_risk_rate_by_category.png', dpi=150)
    plt.close(fig)

    print(f'\nSaved EDA charts in {output_dir}')


if __name__ == '__main__':
    main()