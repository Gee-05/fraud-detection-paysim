import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🔍")

st.title("🔍 Fraud Detection Demo")
st.write("Enter transaction details to get a fraud risk prediction from a random forest model trained on PaySim mobile money data.")

# Load model and feature list
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.divider()

st.subheader("Transaction Details")

amount = st.number_input("Transaction amount", min_value=0.0, value=5000.0, step=100.0)
old_balance_dest = st.number_input("Recipient's balance before transaction", min_value=0.0, value=0.0, step=100.0)
new_balance_dest = st.number_input("Recipient's balance after transaction", min_value=0.0, value=0.0, step=100.0)

st.divider()

if st.button("Predict Fraud Risk", type="primary"):
    # Compute balance_mismatch_dest the same way it was engineered in the notebook
    balance_mismatch_dest = 1 if round(old_balance_dest + amount, 2) != round(new_balance_dest, 2) else 0

    input_data = pd.DataFrame([[amount, old_balance_dest, new_balance_dest, balance_mismatch_dest]],
                                columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **High fraud risk** — predicted probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low fraud risk** — predicted probability: {probability:.1%}")

    st.progress(float(probability))