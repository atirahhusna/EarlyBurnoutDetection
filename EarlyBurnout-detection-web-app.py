import streamlit as st
import pandas as pd
import pickle

# === Load the trained model and encoders ===
with open('trained_model.sav', 'rb') as f:
    loaded_model, feature_columns, label_encoders = pickle.load(f)

# === Question to Feature Mapping ===
questions_map = {
    "What is your Gender?": "Gender",
    "What is your Occupation?": "Occupation",
    "Are you self-employed?": "self_employed",
    "Does any of your family have mental health problems?": "family_history",
    "Have you ever received mental health treatment?": "treatment",
    "How many days do you spend indoors (never go out)?": "Days_Indoors",
    "Do your habits frequently change?": "Changes_Habits",
    "Does your mood always swing?": "Mood_Swings",
    "Are you coping well with struggles?": "Coping_Struggles",
    "Do you still have interest in work?": "Work_Interest",
    "Do you have difficulty socializing with others?": "Social_Weakness",
    "Have you ever attended any mental health interview?": "mental_health_interview",
    "Do you have a mental health therapist available to you?": "care_options"
}

# === Answer Options from Dataset ===
options_per_question = {
    "What is your Gender?": ['Female', 'Male'],
    "What is your Occupation?": ['Business', 'Corporate', 'Housewife', 'Others', 'Student'],
    "Are you self-employed?": ['No', 'Yes'],
    "Does any of your family have mental health problems?": ['No', 'Yes'],
    "Have you ever received mental health treatment?": ['No', 'Yes'],
    "How many days do you spend indoors (never go out)?": [
        '1-14 days', '15-30 days', '31-60 days', 'Go out Every day', 'More than 2 months'
    ],
    "Do your habits frequently change?": ['Maybe', 'No', 'Yes'],
    "Does your mood always swing?": ['High', 'Low', 'Medium'],
    "Are you coping well with struggles?": ['No', 'Yes'],
    "Do you still have interest in work?": ['Maybe', 'No', 'Yes'],
    "Do you have difficulty socializing with others?": ['Maybe', 'No', 'Yes'],
    "Have you ever attended any mental health interview?": ['Maybe', 'No', 'Yes'],
    "Do you have a mental health therapist available to you?": ['No', 'Not sure', 'Yes']
}

# === Streamlit App ===
def main():
    st.title("🧠 Burnout Stress Level Prediction")
    st.markdown("Answer each question below using the dropdown menus.")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.answers = {}

    question_list = list(questions_map.items())

    if st.session_state.step < len(question_list):
        q_text, feature = question_list[st.session_state.step]
        options = options_per_question[q_text]

        st.subheader(f"Q{st.session_state.step + 1}: {q_text}")
        choice = st.selectbox("Select one:", options, key=f"step_{st.session_state.step}")

        if st.button("Next"):
            st.session_state.answers[feature] = choice
            st.session_state.step += 1
            st.experimental_rerun()

    else:
        st.subheader("✅ All questions answered. Predicting stress level...")
        user_input = st.session_state.answers

        # Encode input
        encoded_input = {}
        for feature in feature_columns:
            value = user_input.get(feature, "")
            if feature in label_encoders:
                encoded_input[feature] = label_encoders[feature].transform([value])[0]
            else:
                encoded_input[feature] = value  # For future use

        input_df = pd.DataFrame([encoded_input])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        prediction = loaded_model.predict(input_df)[0]
        if 'Growing_Stress' in label_encoders:
            label = label_encoders['Growing_Stress'].inverse_transform([prediction])[0]
            st.success(f"🌟 Predicted Growing Stress Level: **{label}**")
        else:
            st.success(f"🌟 Predicted Growing Stress Level (encoded): **{prediction}**")

        if st.button("Start Over"):
            st.session_state.step = 0
            st.session_state.answers = {}
            st.experimental_rerun()

if __name__ == '__main__':
    main()
