# Fraud Detection on PaySim Mobile Money Data

A fraud detection system built on PaySim, a synthetic dataset modeling real African mobile
money transaction logs — chosen over the standard Kaggle credit card dataset for stronger
relevance to mobile money fraud (the domain this project is aimed at).

## What this project does

- Loads 6.3M+ transactions into SQLite and engineers fraud-relevant features via SQL
- Investigates and corrects a real data leakage issue found in the raw dataset
- Trains and compares logistic regression vs. random forest on a leakage-free feature set
- Explains model predictions using SHAP
- Provides an interactive Streamlit dashboard for live fraud-risk predictions

## Key finding: data leakage investigation

The first models trained on this dataset scored a suspicious ~1.00 AUC-PR — too good to
trust. Investigation traced this to a known PaySim artifact: origin-account balance fields
are often left unpopulated for legitimate transactions but precisely computed for fraud,
making them act as a disguised copy of the label rather than genuine signals. After removing
all origin-balance-derived features, the model produces an honest, defensible result:

| Model                | Precision (Fraud) | Recall (Fraud) | AUC-PR |
|----------------------|--------------------|-----------------|--------|
| Logistic Regression  | 0.04               | 0.70            | 0.32   |
| Random Forest        | 0.82               | 0.44            | 0.48   |

Full investigation and reasoning in [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb).

## Project structure

```
fraud-detection-paysim/
├── data/           # PaySim dataset (not committed — see below)
├── notebooks/      # Exploration, feature engineering, modeling, SHAP analysis
├── sql/            # Feature engineering queries
├── src/            # Streamlit dashboard (app.py) + saved model
```

## Running this project

1. Download the PaySim dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) into `data/`
2. Install dependencies: `uv pip install pandas scikit-learn matplotlib seaborn jupyter shap streamlit joblib`
3. Run `notebooks/01_exploration.ipynb` to reproduce the analysis and save the model
4. Run the dashboard: `cd src && streamlit run app.py`

## Tech stack

Python, pandas, scikit-learn, SQLite, SHAP, Streamlit