import pandas as pd
import matplotlib.pyplot as plt


def profile(path='data/heart_disease.csv'):
    df = pd.read_csv(path)
    print('Shape:', df.shape)
    print('\nMissing values:\n', df.isna().sum())
    print('\nTarget distribution:\n', df['high_risk'].value_counts(normalize=True).round(3))
    print('\nNumeric summary:\n', df.select_dtypes('number').describe().T)
    return df


def save_target_plot(path='data/heart_disease.csv', output='reports/risk_distribution.png'):
    df = pd.read_csv(path)
    df['high_risk'].value_counts().sort_index().plot(kind='bar')
    plt.title('Healthcare Risk Class Distribution')
    plt.xlabel('High Risk')
    plt.ylabel('Patients')
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

if __name__ == '__main__':
    profile()
