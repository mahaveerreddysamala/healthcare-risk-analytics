from ucimlrepo import fetch_ucirepo


def download(output='data/heart_disease.csv'):
    dataset = fetch_ucirepo(id=45)
    df = dataset.data.original.copy()
    # UCI target is 0 for absence and 1-4 for presence; convert to binary high-risk label.
    target = 'num' if 'num' in df.columns else df.columns[-1]
    df['high_risk'] = (df[target].astype(str).str.strip() != '0').astype(int)
    df.to_csv(output, index=False)
    print(f'Wrote {len(df)} rows to {output}')

if __name__ == '__main__':
    download()
