import streamlit as st
import pandas as pd
import pickle

# === Load model and encoders ===
with open('trained_model.sav', 'rb') as f:
    loaded_model, feature_columns, label_encoders = pickle.load(f)

# === Questions and dataset options ===
questions_map = {
    "What is your Gender?": "Gender",
    "What is your Occupation?": "Occupation",
    "Are you self-employed?": "self_employed",
    "Does any of your family have mental health problems?": "family_history",
    "Have you ever received mental health treatment?": "treatment",
    "Do You Have  mental health history?": "Mental_Health_History",
    "How many days do you spend indoors (never go out)?": "Days_Indoors",
    "Do your habits frequently change?": "Changes_Habits",
    "Does your mood always swing?": "Mood_Swings",
    "Are you coping well with struggles?": "Coping_Struggles",
    "Do you still have interest in work?": "Work_Interest",
    "Do you have difficulty socializing with others?": "Social_Weakness",
    "Have you ever attended any mental health interview?": "mental_health_interview",
    "Do you have a mental health therapist available to you?": "care_options"
}

options_per_question = {
    "What is your Gender?": ['Female', 'Male'],
    "What is your Occupation?": ['Business', 'Corporate', 'Housewife', 'Others', 'Student'],
    "Are you self-employed?": ['No', 'Yes'],
    "Does any of your family have mental health problems?": ['No', 'Yes'],
    "Have you ever received mental health treatment?": ['No', 'Yes'],
    "What is your past mental health history?": ['No', 'Yes'],
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

def main():
    st.title("🧠 Early Burnout Detection")
    st.markdown("Answer each question below using the dropdown menu.")

    # === Initialize session state ===
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.answers = {}

    # Filter only questions for features the model expects
    filtered_questions_map = {
        q: f for q, f in questions_map.items() if f in feature_columns
    }

    filtered_options_per_question = {
        q: options_per_question[q] for q in filtered_questions_map
    }

    question_list = list(filtered_questions_map.items())
    total_steps = len(question_list)

    if st.session_state.step < total_steps:
        q_text, feature = question_list[st.session_state.step]
        options = filtered_options_per_question[q_text]

        st.subheader(f"Q{st.session_state.step + 1} of {total_steps}: {q_text}")

        # Show previous answer if exists
        previous_answer = st.session_state.answers.get(feature, options[0])
        choice = st.selectbox("Select one:", options, key=f"step_{st.session_state.step}", index=options.index(previous_answer))

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅️ Back"):
                st.session_state.step = max(0, st.session_state.step - 1)
                st.rerun()

        with col2:
            if st.button("➡️ Next"):
                st.session_state.answers[feature] = choice
                st.session_state.step += 1
                st.rerun()

    else:
        st.subheader("✅ All questions answered. Predicting stress level...")

        user_input = st.session_state.answers
        encoded_input = {}

        for feature in feature_columns:
            if feature in user_input:
                value = user_input[feature]
                if feature in label_encoders:
                    try:
                        encoded_input[feature] = label_encoders[feature].transform([value])[0]
                    except Exception:
                        st.error(f"⚠️ Invalid input for feature `{feature}`: {value}")
                        return
                else:
                    encoded_input[feature] = value
            else:
                st.error(f"❌ Missing required input for feature: {feature}")
                return

        input_df = pd.DataFrame([encoded_input])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        try:
            prediction = loaded_model.predict(input_df)[0]

            if 'Growing_Stress' in label_encoders:
                label = label_encoders['Growing_Stress'].inverse_transform([prediction])[0]
                st.success(f"🌟 Predicted Growing Stress: **{label}**")

                if label == "Yes":
                    st.warning("🚨 Your stress level is likely **growing**. Please seek professional help.")
                    st.markdown(
                        "🔗 Visit [Mental Health Services - Ministry of Health Malaysia (MOH)](https://www.moh.gov.my/) "
                        "for support and nearby resources."
                    )
                else:
                    st.info("😊 Your mental health appears to be stable. Keep taking care of yourself!")
            else:
                st.success(f"🌟 Predicted Growing Stress Level (encoded): **{prediction}**")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

        if st.button("🔄 Start Over"):
            st.session_state.step = 0
            st.session_state.answers = {}
            st.rerun()

if __name__ == '__main__':
    main()
