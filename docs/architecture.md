# Healthcare Risk Architecture

```mermaid
flowchart LR
  A[UCI Heart Disease Data] --> B[Data Validation]
  B --> C[Clinical Feature Engineering]
  C --> D[Train/Test Split]
  D --> E[Logistic Regression]
  D --> F[Random Forest]
  E --> G[Evaluation]
  F --> G
  G --> H[Risk Probability]
  H --> I[Streamlit Analytics]
```

This is an educational analytics project and must not be presented as a clinical decision-support system. Model outputs require independent validation before any real-world healthcare use.
