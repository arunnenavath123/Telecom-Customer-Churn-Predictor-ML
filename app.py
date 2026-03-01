import streamlit as st
import pandas as pd
import joblib
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# --- Load Model and Artifacts ---
model = joblib.load("xgb_churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- Initialize OpenAI Client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# --- Streamlit UI ---
st.title("📊 Telecom Customer Churn Predictor")

# --- User Input ---
def user_input_features():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    return pd.DataFrame([data])

# --- Prompt Generation for LLM Explanation ---
def generate_prompt(original_df, encoded_df, prediction):
    row = original_df.iloc[0]
    encoded = encoded_df.iloc[0]
    prediction_label = "will churn" if prediction == 1 else "will not churn"

    prompt = f"""
You are a data analyst helping to explain churn prediction for a telecom customer using their profile. The model predicted that the customer **{prediction_label}**.

Below is the customer's profile:
- Gender: {row['gender']}
- Senior Citizen: {'Yes' if row['SeniorCitizen'] == 1 else 'No'}
- Has Partner: {row['Partner']}
- Has Dependents: {row['Dependents']}
- Tenure: {row['tenure']} months
- Phone Service: {row['PhoneService']}
- Multiple Lines: {row['MultipleLines']}
- Internet Service: {row['InternetService']}
- Online Security: {row['OnlineSecurity']}
- Online Backup: {row['OnlineBackup']}
- Device Protection: {row['DeviceProtection']}
- Tech Support: {row['TechSupport']}
- Streaming TV: {row['StreamingTV']}
- Streaming Movies: {row['StreamingMovies']}
- Contract Type: {row['Contract']}
- Paperless Billing: {row['PaperlessBilling']}
- Payment Method: {row['PaymentMethod']}
- Monthly Charges: {row['MonthlyCharges']}
- Total Charges: {row['TotalCharges']}

Based on this profile, explain in plain language **why the model predicted that the customer {prediction_label}**. Highlight any key factors that may have influenced the decision, such as tenure, contract type, internet service, or payment method. Avoid making assumptions not reflected in the data.
"""
    return prompt

# --- Call LLM ---
def get_llm_explanation(prompt):
    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return completion.choices[0].message.content

# --- Main Prediction Logic ---
input_df_raw = user_input_features()

if st.button("Predict Churn"):
    try:
        # Encode categorical features using label encoders
        for col in label_encoders:
            input_df_raw[col] = label_encoders[col].transform([input_df_raw[col].values[0]])

        # One-hot encode remaining categorical columns
        cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']
        input_df_encoded = pd.get_dummies(input_df_raw, columns=cat_cols, drop_first=True)

        # Align columns with training
        input_df_encoded = input_df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Ensure numeric fields are clean
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            input_df_encoded[col] = pd.to_numeric(input_df_encoded[col], errors='coerce')

        # Predict
        prediction = model.predict(input_df_encoded)[0]
        prediction_proba = model.predict_proba(input_df_encoded)[0][1]

        st.write("### Telecom Customer Churn Prediction")
        st.write(f"🔮 Will the customer churn? {'Yes' if prediction == 1 else 'No'}")
        st.write(f"Prediction confidence: {prediction_proba:.2f}")

        with st.spinner("Generating explanation..."):
            prompt = generate_prompt(input_df_raw, input_df_encoded, prediction)
            explanation = get_llm_explanation(prompt)
            st.write("🧠 AI-generated explanation (LLM)")
            st.write(explanation)

        with st.expander("🔍 View Processed Input Data"):
            st.dataframe(input_df_encoded)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and XGBoost")
