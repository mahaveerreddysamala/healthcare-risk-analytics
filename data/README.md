# Dataset

The project supports the UCI Heart Disease dataset (ID 45). UCI reports 303 instances and 13 features for the commonly used processed dataset, with categorical, integer and continuous variables and some missing values. The target represents presence/absence of heart disease; this project maps non-zero target values to a binary `high_risk` label.

The data loader in `src/download_data.py` uses the `ucimlrepo` package so the public dataset can be retrieved reproducibly instead of storing the source data in Git.

Dataset citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. DOI: 10.24432/C52P4X.
