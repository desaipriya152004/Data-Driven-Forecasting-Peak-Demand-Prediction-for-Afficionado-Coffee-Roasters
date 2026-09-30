# ☕ Afficionado Coffee Roasters — Streamlit Analytics Portfolio

An interactive portfolio web app built from the supplied Afficionado Coffee Roasters project:
cleaned transaction data, EDA/forecasting notebook, Power BI PBIX, dashboard screenshots and EDA report.

## What the app contains

- **Home** — portfolio landing page and project story
- **Executive Overview** — revenue, quantity, transaction and store/category analysis
- **Peak Demand & Operations** — hourly demand, store-vs-hour heatmap and peak-hour table
- **Product & Store Analysis** — category mix, quantity/revenue relationships and store comparison
- **Forecasting & Demand Planning** — interactive demand prediction using the supplied HistGradientBoostingRegressor
- **EDA & Data Explorer** — filter and inspect cleaned transaction-level data
- **Power BI Reference** — visual reference of the original five-page PBIX dashboard

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload this project.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Choose **Create app**, select the repository, branch and `app.py`.
4. Deploy.

Streamlit Community Cloud reads the dependency list from `requirements.txt`.

## Project note

The supplied source has a year and time-of-day but no genuine calendar transaction date. The original notebook creates a transaction date from year + time, which results in 2025-01-01 for the supplied records. Therefore the model is presented as an **intraday prediction baseline**, not a validated multi-day/seasonal forecasting system.

The original notebook reports:
- MAE: **0.4365**
- RMSE: **0.4899**
- Train/test split: **80% / 20%**
- Model: **HistGradientBoostingRegressor**
