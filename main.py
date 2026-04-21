import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

st.set_page_config(page_title="AKI Predictor", layout="centered")
st.title("🏥 Post-CABG AKI Risk Predictor")

model = xgb.XGBClassifier()
model.load_model('model.json')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

feature_columns = ['age', 'gender', 'baseline_val', 'heart_rate', 'resprate', 'mean_map', 'temp_c', 'spo2', 'glucose', 'lactate', 'ph', 'pco2', 'po2', 'bicarbonate', 'chloride', 'potassium', 'sodium', 'hematocrit', 'hemoglobin', 'platelets', 'wbc', 'bun']

with st.form("patient_form"):
    # ТУТ НУЖНО ДОБАВИТЬ ОТСТУПЫ (4 ПРОБЕЛА) ПЕРЕД СТРОКАМИ НИЖЕ:
    st.subheader("Patient Clinical Data")
    age = st.number_input("Age", 18, 100, 65)
    gender = st.selectbox("Gender", ["M", "F"])
    bun = st.number_input("BUN Level", 0.0, 150.0, 20.0)
    baseline = st.number_input("Baseline Creatinine", 0.0, 10.0, 1.0)
    map_val = st.number_input("MAP (Blood Pressure)", 30, 150, 80)
    lactate = st.number_input("Lactate", 0.0, 20.0, 1.5)
    submit = st.form_submit_button("Run Prediction")

if submit:
    # ТУТ ТОЖЕ НУЖНО ДОБАВИТЬ ОТСТУПЫ (4 ПРОБЕЛА) ПЕРЕД СТРОКАМИ НИЖЕ:
    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    input_df.at[0, 'age'] = age
    input_df.at[0, 'gender'] = 1 if gender == "M" else 0
    input_df.at[0, 'bun'] = bun
    input_df.at[0, 'baseline_val'] = baseline
    input_df.at[0, 'mean_map'] = map_val
    input_df.at[0, 'lactate'] = lactate
    processed = scaler.transform(imputer.transform(input_df))
    prob = model.predict_proba(processed)[0][1]
    st.divider()
    st.subheader(f"Risk: {prob*100:.1f}%")
    if prob > 0.4:
        st.error("High Risk of AKI")
    else:
        st.success("Low Risk")
