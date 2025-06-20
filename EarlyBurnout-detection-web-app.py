# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:27:51 2025

@author: ASUS
"""

import streamlit as st
import pandas as pd
import pickle

# === Load model, feature columns, and label encoders ===
with open('trained_model.sav', 'wb') as f:
    pickle.dump((model, X.columns.tolist(), label_encoders), f)


# === Streamlit App ===
def main():
    st.title("🧠 Burnout Stress Level Prediction App")
    st.markdown("Enter the user details to predict **Growing Stress Level**")

    user_input = {}
    for feature in feature_columns:
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            user_input[feature] = st.selectbox(f"{feature}", options)
        else:
            user_input[feature] = st.number_input(f"{feature}", step=1.0)

    # Predict when button is clicked
    if st.button("Predict"):
        # Encode input
        encoded_input = {}
        for feature in feature_columns:
            if feature in label_encoders:
                encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
            else:
                encoded_input[feature] = user_input[feature]

        input_df = pd.DataFrame([encoded_input])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        pred = loaded_model.predict(input_df)[0]

        if 'Growing_Stress' in label_encoders:
            label = label_encoders['Growing_Stress'].inverse_transform([pred])[0]
            st.success(f"🌟 Predicted Growing Stress Level: **{label}**")
        else:
            st.success(f"🌟 Predicted Growing Stress Level (encoded): **{pred}**")

if __name__ == '__main__':
    main()
