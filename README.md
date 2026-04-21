# AKI Prediction Model after CABG Surgery (MIMIC-IV)

## Project Overview
This project focuses on predicting **Acute Kidney Injury (AKI)** in patients following Coronary Artery Bypass Grafting (CABG). Utilizing the **MIMIC-IV** dataset, I developed a machine learning pipeline to identify high-risk patients early.

## Performance Metrics
- **ROC-AUC Score:** 0.755
- **PR-AUC Score:** [Вставь свою цифру из Colab]
- **Recall (Sensitivity):** 0.46

## Methodology
- **Algorithm:** XGBoost (Extreme Gradient Boosting) with `scale_pos_weight` for class imbalance.
- **Features:** 22 clinical variables including BUN, MAP, and Lactate levels.
- **Preprocessing:** Median Imputation and Standard Scaling.

## Data Ethics
All data used in this project is de-identified and was accessed via PhysioNet. The study adheres to ethical guidelines for clinical data research.

## How to use
The live application is hosted on **Streamlit Cloud**. Use the sidebar to enter clinical parameters and assess the risk probability.

