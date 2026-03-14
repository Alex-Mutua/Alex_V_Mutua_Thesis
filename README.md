<div align="center">

# 📈 Support Vector Machines for Financial Time Series Forecasting
### Models, Techniques, and Performance Evaluation Across Financial Markets

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Thesis](https://img.shields.io/badge/Type-MSc%20Thesis-blueviolet?style=for-the-badge)](#)

---

**Alex V Mutua**

📧 mutua.v.alex@aims-senegal.org

African Institute for Mathematical Sciences (AIMS) — Senegal

🎓 Master of Science in Data Science

</div>

---

# 📋 Abstract

Financial markets generate highly volatile and nonlinear time series that pose significant challenges for forecasting models. Classical econometric approaches such as ARIMA often struggle with nonlinear dependencies, while modern deep learning models require large datasets and substantial computational resources.

This thesis investigates the effectiveness of **Support Vector Regression (SVR)** for forecasting financial time series. Using a unified rolling-window framework, we compare SVR against classical benchmarks (ARIMA, Naive) and machine learning models (LSTM, MLP, XGBoost) across heterogeneous financial markets including **cryptocurrencies, equities, and commodities**.

Our results demonstrate that **tuned RBF-SVR achieves strong forecasting performance and robust directional accuracy**, offering a computationally efficient alternative to deep learning methods for moderate-sized financial datasets.

> Keywords: Support Vector Regression · Financial Forecasting · Machine Learning · Time Series · Rolling Window Evaluation

---

# 📑 Table of Contents

- Introduction
- Background and Literature Review
- Methodology
- Experimental Design
- Results
- Discussion
- Conclusion and Future Work

---

# 🎯 Research Objective

The central goal of this study is to evaluate whether **Support Vector Regression provides a robust and interpretable alternative to deep learning models for financial time series forecasting**.

Key questions addressed include:

1. How does SVR perform relative to classical econometric models?
2. How sensitive is SVR performance to kernel choice and hyperparameters?
3. Can SVR generalize across different financial asset classes?
4. What trade-offs exist between SVR and deep learning models?

---

# 📊 Dataset

The study evaluates forecasting performance across three representative financial markets:

| Asset | Market Type | Ticker |
|------|-------------|--------|
| Bitcoin | Cryptocurrency | BTC-USD |
| S&P 500 | Equity Index | ^GSPC |
| Gold ETF | Commodity | GLD |

These datasets capture **diverse statistical regimes**, allowing assessment of model robustness across heterogeneous financial environments.

---

# ⚙️ Methodology

The forecasting framework follows four key steps:

1️⃣ **Feature Engineering**

- Log returns  
- Lagged returns  
- Rolling volatility  
- Momentum indicators  
- Wavelet decomposition

2️⃣ **Forecasting Models**

- Support Vector Regression (SVR)
- ARIMA
- Naive benchmark
- LSTM
- MLP
- XGBoost

3️⃣ **Evaluation Framework**

A **rolling-window evaluation** is used to simulate real-world forecasting:

Train → Predict next day → Shift window → Retrain

4️⃣ **Performance Metrics**

| Metric | Purpose |
|------|---------|
MAE | Mean absolute forecasting error |
RMSE | Penalizes large errors |
SMAPE | Scale-independent comparison |
Directional Accuracy | Measures correct market direction |

---

# 📈 Key Results

| Model | MAE | RMSE | Directional Accuracy |
|------|------|------|------|
SVR (RBF tuned) | **Best SVR performance** | Competitive | High |
ARIMA | Moderate | Moderate | Lower |
Naive | Weak | Weak | Random baseline |
LSTM | Strong accuracy | High cost | High |
XGBoost | Competitive | Moderate | High |

Key observations:

- Kernel choice significantly affects SVR performance
- Hyperparameter tuning is critical
- SVR performs consistently across heterogeneous assets
- SVR offers strong performance with lower computational cost than deep learning

---

# 📊 Example Forecast

Example forecast comparison for **Bitcoin returns**:

<p align="center">
<img src="figures/btc_forecast.png" width="700">
</p>

---

# 📂 Repository Structure
