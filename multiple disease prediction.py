# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:43:28 2025
@author: Viraj
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="🩺 Multiple Disease Prediction",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- Load Models ---------------------
diabetes_model = pickle.load(open(
    'diabetes_model.sav', 'rb'))

heart_model = pickle.load(open(
    'heartdisease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(
    'parkinsons_model.sav', 'rb'))

parkinsons_scaler = pickle.load(open(
    'parkinsons_scaler.sav', 'rb'))

liver_model = pickle.load(open(
    'liver_model.sav', 'rb'))

liver_scaler = pickle.load(open(
    'liver_scaler.sav', 'rb'))

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; color:#2e86de;'>🩺 Select Disease for Prediction</h2>",
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Liver Disease Prediction'],
        icons=['activity', 'heart', 'person', 'droplet'],
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#ffffff"},
            "icon": {"color": "#2e86de", "font-size": "25px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "left",
                "margin": "5px",
                "color": "#000000"
            },
            "nav-link-selected": {
                "background-color": "#2e86de",
                "color": "white"
            },
        }
    )

# --------------------- Header ---------------------
st.markdown(
    """
    <div style="background: linear-gradient(to right, #2e86de, #6c5ce7);
                padding: 25px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
        <h1 style="color: white;">🩺 Multiple Disease Prediction System</h1>
        <p style="color: white; font-size:16px;">Predict Diabetes, Heart Disease, Parkinson's, and Liver Disease</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------- Helper Function ---------------------
def result_message(prediction, disease_name):
    if prediction[0] == 1:
        st.error(f"⚠️ The Person has **{disease_name}**")
    else:
        st.success(f"✅ The Person has **No {disease_name}**")

# --------------------- Diabetes Prediction ---------------------
if selected == 'Diabetes Prediction':
    st.markdown(
        "<div style='background-color:#3498db; padding:15px; border-radius:10px; color:white; text-align:center;'>"
        "<h2>🩸 Diabetes Prediction</h2></div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200)
        Insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
        Age = st.number_input("Age", min_value=1, max_value=120)
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100)
        BMI = st.number_input("BMI Value", min_value=0.0, max_value=70.0, format="%.1f")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Get Diabetes Test Result"):
        diab_proba = diabetes_model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                               Insulin, BMI, DiabetesPedigreeFunction, Age]])[0][1]
        risk_percent = round(diab_proba * 100, 2)
        st.metric(label="Diabetes Risk", value=f"{risk_percent}%")
        
        if risk_percent > 50:
            st.error(f"⚠️ High Risk of Diabetes ({risk_percent}%)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({risk_percent}%)")
    # ------------------- Download Report -------------------
        report = f"""
        Diabetes Prediction Report
        ---------------------------
        Pregnancies: {Pregnancies}
        Glucose: {Glucose}
        Blood Pressure: {BloodPressure}
        Skin Thickness: {SkinThickness}
        Insulin: {Insulin}
        BMI: {BMI}
        Diabetes Pedigree Function: {DiabetesPedigreeFunction}
        Age: {Age}
        
        Risk Percentage: {risk_percent}%
        Result: {'High Risk' if risk_percent > 50 else 'Low Risk'}
        """
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="diabetes_report.txt",
            mime="text/plain"
        )

# --------------------- Heart Disease Prediction ---------------------
if selected == 'Heart Disease Prediction':
    st.markdown(
        "<div style='background-color:#e74c3c; padding:15px; border-radius:10px; color:white; text-align:center;'>"
        "<h2>❤️ Heart Disease Prediction</h2></div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
        cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=250)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=0, max_value=600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    with col2:
        restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250)
        exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, format="%.1f")
        slope = st.number_input("Slope of ST Segment (0-2)", min_value=0, max_value=2)
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
        thal = st.number_input("Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Other)", min_value=0, max_value=3)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Get Heart Disease Test Result"):
        heart_proba = heart_model.predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg,
                                             thalach, exang, oldpeak, slope, ca, thal]])[0][1]
        risk_percent = round(heart_proba * 100, 2)
        st.metric(label="Heart Disease Risk", value=f"{risk_percent}%")
        
        if risk_percent > 50:
            st.error(f"⚠️ High Risk of Heart Disease ({risk_percent}%)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({risk_percent}%)")
        report = f"""
        Heart Disease Prediction Report
        -------------------------------
        Age: {age}
        Sex: {'Male' if sex == 1 else 'Female'}
        Chest Pain Type: {cp}
        Resting Blood Pressure: {trestbps}
        Serum Cholesterol: {chol}
        Fasting Blood Sugar > 120 mg/dl: {'Yes' if fbs == 1 else 'No'}
        Resting ECG: {restecg}
        Max Heart Rate Achieved: {thalach}
        Exercise Induced Angina: {'Yes' if exang == 1 else 'No'}
        ST Depression: {oldpeak}
        Slope of ST Segment: {slope}
        Number of Major Vessels: {ca}
        Thal: {thal}
        
        Risk Percentage: {risk_percent}%
        Result: {'High Risk' if risk_percent > 50 else 'Low Risk'}
        """
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="heart_report.txt",
            mime="text/plain"
        )

# --------------------- Parkinson’s Prediction ---------------------
if selected == 'Parkinsons Prediction':
    st.markdown(
        "<div style='background-color:#27ae60; padding:15px; border-radius:10px; color:white; text-align:center;'>"
        "<h2>🧠 Parkinson's Disease Prediction</h2></div>", unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input("MDVP:Fo(Hz)", "0.0")
        fhi = st.text_input("MDVP:Fhi(Hz)", "0.0")
        flo = st.text_input("MDVP:Flo(Hz)", "0.0")
        jitter_percent = st.text_input("MDVP:Jitter(%)", "0.0")
        jitter_abs = st.text_input("MDVP:Jitter(Abs)", "0.0")
        rap = st.text_input("MDVP:RAP", "0.0")
        ppq = st.text_input("MDVP:PPQ", "0.0")
    with col2:
        ddp = st.text_input("Jitter:DDP", "0.0")
        shimmer = st.text_input("MDVP:Shimmer", "0.0")
        shimmer_db = st.text_input("MDVP:Shimmer(dB)", "0.0")
        apq3 = st.text_input("Shimmer:APQ3", "0.0")
        apq5 = st.text_input("Shimmer:APQ5", "0.0")
        apq = st.text_input("MDVP:APQ", "0.0")
        dda = st.text_input("Shimmer:DDA", "0.0")
    with col3:
        nhr = st.text_input("NHR", "0.0")
        hnr = st.text_input("HNR", "0.0")
        rpde = st.text_input("RPDE", "0.0")
        dfa = st.text_input("DFA", "0.0")
        spread1 = st.text_input("Spread1", "0.0")
        spread2 = st.text_input("Spread2", "0.0")
        d2 = st.text_input("D2", "0.0")
        ppe = st.text_input("PPE", "0.0")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Get Parkinson's Test Result"):
        try:
            features = [
                float(fo), float(fhi), float(flo), float(jitter_percent), float(jitter_abs),
                float(rap), float(ppq), float(ddp), float(shimmer), float(shimmer_db),
                float(apq3), float(apq5), float(apq), float(dda), float(nhr), float(hnr),
                float(rpde), float(dfa), float(spread1), float(spread2), float(d2), float(ppe)
            ]
            features_scaled = parkinsons_scaler.transform([features])
            parkinsons_proba = parkinsons_model.predict_proba(features_scaled)[0][1]
            risk_percent = round(parkinsons_proba * 100, 2)
            st.metric(label="Parkinson's Disease Risk", value=f"{risk_percent}%")
            
            if risk_percent > 50:
                st.error(f"⚠️ High Risk of Parkinson's ({risk_percent}%)")
            else:
                st.success(f"✅ Low Risk of Parkinson's ({risk_percent}%)")
            
            # ------------------- Download Report -------------------
            report = f"""
    Parkinson's Disease Prediction Report
    -------------------------------------
    MDVP:Fo(Hz): {fo}
    MDVP:Fhi(Hz): {fhi}
    MDVP:Flo(Hz): {flo}
    MDVP:Jitter(%): {jitter_percent}
    MDVP:Jitter(Abs): {jitter_abs}
    MDVP:RAP: {rap}
    MDVP:PPQ: {ppq}
    Jitter:DDP: {ddp}
    MDVP:Shimmer: {shimmer}
    MDVP:Shimmer(dB): {shimmer_db}
    Shimmer:APQ3: {apq3}
    Shimmer:APQ5: {apq5}
    MDVP:APQ: {apq}
    Shimmer:DDA: {dda}
    NHR: {nhr}
    HNR: {hnr}
    RPDE: {rpde}
    DFA: {dfa}
    Spread1: {spread1}
    Spread2: {spread2}
    D2: {d2}
    PPE: {ppe}
    
    Risk Percentage: {risk_percent}%
    Result: {'High Risk' if risk_percent > 50 else 'Low Risk'}
    """
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="parkinsons_report.txt",
                mime="text/plain"
            )
    
        except ValueError:
            st.warning("⚠️ Please enter valid numeric values for all fields.")


# --------------------- Liver Disease Prediction ---------------------
if selected == 'Liver Disease Prediction':
    st.markdown(
        "<div style='background-color:#8e44ad; padding:15px; border-radius:10px; color:white; text-align:center;'>"
        "<h2>🧪 Liver Disease Prediction</h2></div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=100.0, format="%.2f")
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=50.0, format="%.2f")
        alkphos = st.number_input("Alkaline Phosphotase", min_value=0.0, max_value=2000.0)
        sgpt = st.number_input("SGPT (Alanine Aminotransferase)", min_value=0.0, max_value=2000.0)
    with col2:
        sgot = st.number_input("SGOT (Aspartate Aminotransferase)", min_value=0.0, max_value=2000.0)
        total_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=800.0, format="%.2f")
        albumin = st.number_input("Albumin", min_value=0.0, max_value=300.0, format="%.2f")
        ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=500.0, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Get Liver Disease Test Result"):
        try:
            # Create feature vector
            features = [[
                age, gender_val, total_bilirubin, direct_bilirubin, alkphos,
                sgpt, sgot, total_proteins, albumin, ag_ratio
            ]]
    
            # Scale features
            features_scaled = liver_scaler.transform(features)
    
            # Prediction & probability
            proba = liver_model.predict_proba(features_scaled)[0][1]
            risk_percent = round(proba * 100, 2)
            prediction = liver_model.predict(features_scaled)[0]
    
            # Show results
            st.metric(label="Liver Disease Risk", value=f"{risk_percent}%")
    
            if prediction == 1:
                st.error(f"⚠️ High Risk of Liver Disease ({risk_percent}%)")
            else:
                st.success(f"✅ Low Risk of Liver Disease ({risk_percent}%)")
            
            report = f"""
            Liver Disease Prediction Report
            -------------------------------
            Age: {age}
            Gender: {'Male' if gender_val == 1 else 'Female'}
            Total Bilirubin: {total_bilirubin}
            Direct Bilirubin: {direct_bilirubin}
            Alkaline Phosphotase: {alkphos}
            SGPT (Alanine Aminotransferase): {sgpt}
            SGOT (Aspartate Aminotransferase): {sgot}
            Total Proteins: {total_proteins}
            Albumin: {albumin}
            Albumin and Globulin Ratio: {ag_ratio}
            
            Risk Percentage: {risk_percent}%
            Result: {'High Risk' if prediction == 1 else 'Low Risk'}
            """
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="liver_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.warning(f"⚠️ Error: {e}")
       

