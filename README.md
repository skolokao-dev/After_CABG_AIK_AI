# AKI Prediction Model after CABG Surgery (MIMIC-IV, eICU, INSPIRE)

## Project Overview
This project focuses on predicting **Acute Kidney Injury (AKI)** in patients following Coronary Artery Bypass Grafting (CABG). Utilizing the MIMIC-IV, eICU, INSPIRE datasets, I developed a machine learning pipeline to identify high-risk patients early.

## Performance Metrics
- **ROC-AUC Score:** 0.7902
- **Recall (Sensitivity):** 0.6765

## Methodology
- **Algorithm:** XGBoost (Extreme Gradient Boosting) with `scale_pos_weight` for class imbalance.
- **Features:** 22 clinical variables including BUN, MAP, and Lactate levels.
- **Preprocessing:** Median Imputation and Standard Scaling.

## Data Ethics
All data used in this project is de-identified and was accessed via PhysioNet. The study adheres to ethical guidelines for clinical data research.

## How to use
The live application is hosted on **Streamlit Cloud**. Use the sidebar to enter clinical parameters and assess the risk probability.

