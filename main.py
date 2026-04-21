import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

st.set_page_config(page_title="Full AKI Predictor", layout="wide")
st.title("🏥 Clinical Post-CABG AKI Predictor")

model = xgb.XGBClassifier()
model.load_model('model.json')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

with st.form("full_patient_form"):
    st.subheader("1. General & Vitals")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 65)
        gender = st.selectbox("Gender", ["M", "F"])
        hr = st.number_input("Heart Rate (bpm)", 30, 200, 80)
    with c2:
        map_val = st.number_input("MAP (mmHg)", 30, 150, 80)
        temp = st.number_input("Temp (°C)", 34.0, 42.0, 36.6)
        resp = st.number_input("Resp Rate", 10, 50, 18)
    with c3:
        spo2 = st.number_input("SPO2 (%)", 50, 100, 98)
        baseline = st.number_input("Baseline Creatinine", 0.1, 10.0, 1.0)

    st.subheader("2. Blood Gases & Chemistry")
    c4, c5, c6 = st.columns(3)
    with c4:
        lactate = st.number_input("Lactate", 0.0, 20.0, 1.5)
        ph = st.number_input("pH", 6.8, 7.8, 7.4)
        pco2 = st.number_input("pCO2", 10, 100, 40)
    with c5:
        po2 = st.number_input("pO2", 20, 500, 90)
        bicarb = st.number_input("Bicarbonate", 5, 50, 24)
        sodium = st.number_input("Sodium", 100, 160, 140)
    with c6:
        potas = st.number_input("Potassium", 2.0, 8.0, 4.0)
        chlor = st.number_input("Chloride", 70, 130, 104)
        glu = st.number_input("Glucose", 50, 500, 110)

    st.subheader("3. Hematology & Renal")
    c7, c8, c9 = st.columns(3)
    with c7:
        hemato = st.number_input("Hematocrit", 10, 60, 40)
        hemog = st.number_input("Hemoglobin", 5, 20, 13)
    with c8:
        plat = st.number_input("Platelets", 10, 1000, 250)
        wbc = st.number_input("WBC", 1, 50, 10)
    with c9:
        bun = st.number_input("BUN Level", 1, 150, 20)

    submit = st.form_submit_button("Calculate Full Risk Score")

if submit:
    # Передаем данные списком, чтобы избежать проблем с именами колонок
    raw_data = [age, 1 if gender == "M" else 0, baseline, hr, resp, map_val, temp, spo2, glu, lactate, ph, pco2, po2, bicarb, chlor, potas, sodium, hemato, hemog, plat, wbc, bun]
    input_array = np.array([raw_data])
    
    # Сначала заполняем пропуски (если есть), потом масштабируем
    imputed = imputer.transform(input_array)
    processed = scaler.transform(imputed)
    
    # Предсказание
    prob = model.predict_proba(processed)[0][1]
    
    st.divider()
    st.subheader(f"Risk Probability: {prob*100:.1f}%")
    if prob > 0.4:
        st.error("HIGH RISK: Intensive monitoring recommended.")
    else:
        st.success("LOW RISK: Within expected postoperative range.")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** For research purposes only. Not medical advice.")
