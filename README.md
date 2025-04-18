![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pb RMSE](https://img.shields.io/badge/Pb%20RMSE-___-orange)
![Pb R²](https://img.shields.io/badge/Pb%20R²-___-brightgreen)
![Bob RMSE](https://img.shields.io/badge/Bob%20RMSE-___-orange)
![Bob R²](https://img.shields.io/badge/Bob%20R²-___-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/boushboo/pvt-prediction-ml-)



# PVT Prediction Using Machine Learning

This project focuses on predicting **bubble point pressure (Pb)** and **oil formation volume factor (Bob)** using various machine learning models trained on real reservoir data from multiple wells.

The aim is to develop accurate, efficient, and generalizable models that outperform traditional empirical correlations and support better decision-making in reservoir engineering.

---

## 📁 Project Structure
```
pvt-prediction-ml/
├── Dataset/           ← Raw datasets (excluded from GitHub)
├── Phase 1/           ← Notebooks for simple baseline models
├── Phase 2/           ← Notebooks for more complex ML models
├── README.md
└── .gitignore
```


---

## 🧪 Approach Overview

The modeling process is structured into **two main phases**, each applied across **three wells** with varying sample sizes:

- **Well 1:** 160 samples  
- **Well 2:** 283 samples  
- **Well 3:** 782 samples
- 🔄 Combined Dataset: 1225 samples
(Used to test generalization performance across all wells)

**🔸 Input Features (Predictors):**
- `Tf` – Reservoir Temperature (°F)
- `Rs` – Solution Gas-Oil Ratio (scf/STB)
- `gg` – Gas Gravity
- `api` – Oil API Gravity

**🔹 Target Variables (Labels):**
- `Pb` – Bubble Point Pressure (psi)
- `Bob` – Oil Formation Volume Factor (bbl/STB)

### 🔹 Phase 1 — Baseline Models
Explores simple ML models:
- Linear Regression
- Ridge, Lasso
- Decision Tree
- KNN
- Support Vector Regression

### 🔹 Phase 2 — Advanced Models
Focuses on:
- XGBoost
- CatBoost
- Neural Networks
- Stacked Ensembles
- Mixture of Experts
- Log-transformation
- Feature importance and ablation
- Outlier handling

---

## 📊 Model Evaluation

Each model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Adjusted R² Score
- 5-Fold Cross-Validation

Visualizations include:
- Actual vs Predicted scatter plots
- Bar plots comparing metrics
- Feature importance plots

---

## 🔐 Data Privacy

The raw reservoir datasets are **not included** in this repository for privacy reasons.  
If you're reproducing this project:
- Use your own data with similar structure
- Or contact us to request access (if applicable)

---

## 🚀 Running the Project

1. Clone the repo:

```bash
git clone https://github.com/your-username/pvt-prediction-ml.git
