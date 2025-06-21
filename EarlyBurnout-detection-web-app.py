import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model, feature columns, and label encoders
with open('trained_model.sav', 'rb') as f:
    loaded_model, feature_columns, label_encoders = pickle.load(f)

# Streamlit App
def main():
    st.title("🧠 Burnout Stress Level Prediction App")
    st.markdown("Enter the user details to predict **Growing Stress Level**")
    
    # Create input fields manually for each feature
    gender = st.selectbox("What is your Gender?", label_encoders['Gender'].classes_)
    occupation = st.selectbox("What is your Occupation?", label_encoders['Occupation'].classes_)
    self_employed = st.selectbox("Are you self-employed?", label_encoders['self_employed'].classes_)
    family_history = st.selectbox("Does any of your family have mental health problems?", label_encoders['family_history'].classes_)
    treatment = st.selectbox("Have you ever received mental health treatment?", label_encoders['treatment'].classes_)
    days_indoors = st.number_input("How many days do you spend indoors (never go out)?", step=1.0)
    changing_habits = st.selectbox("Do your habits frequently change?", label_encoders['changing_habits'].classes_)
    mood_swings = st.selectbox("Does your mood always swing?", label_encoders['mood_swings'].classes_)
    coping_struggles = st.selectbox("Are you coping well with struggles?", label_encoders['coping_struggles'].classes_)
    interest_to_work = st.selectbox("Do you still have interest in work?", label_encoders['interest_to_work'].classes_)
    difficulty_socializing = st.selectbox("Do you have difficulty socializing with others?", label_encoders['difficulty_socializing'].classes_)
    interview = st.selectbox("Have you ever attended any mental health interview?", label_encoders['interview'].classes_)
    availability = st.selectbox("Do you have a mental health therapist available to you?", label_encoders['availability'].classes_)

    # Predict when button is clicked
    if st.button("Predict"):
        # Create a dictionary with all inputs
        input_data = {
            'Gender': gender,
            'Occupation': occupation,
            'self_employed': self_employed,
            'family_history': family_history,
            'treatment': treatment,
            'Days_indoors': days_indoors,
            'changing_habits': changing_habits,
            'mood_swings': mood_swings,
            'coping_struggles': coping_struggles,
            'interest_to_work': interest_to_work,
            'difficulty_socializing': difficulty_socializing,
            'interview': interview,
            'availability': availability
        }
        
        # Encode categorical features
        encoded_data = {}
        for feature, value in input_data.items():
            if feature in label_encoders:
                encoded_data[feature] = label_encoders[feature].transform([value])[0]
            else:
                encoded_data[feature] = value
        
        # Create DataFrame in correct feature order
        input_df = pd.DataFrame([encoded_data], columns=feature_columns)
        
        # Make prediction
        pred = loaded_model.predict(input_df)[0]
        
        # Decode prediction if label encoder exists
        if 'Growing_Stress' in label_encoders:
            label = label_encoders['Growing_Stress'].inverse_transform([pred])[0]
            st.success(f"🌟 Predicted Growing Stress Level: **{label}**")
        else:
            st.success(f"🌟 Predicted Growing Stress Level (encoded): **{pred}**")

if __name__ == '__main__':
    main()