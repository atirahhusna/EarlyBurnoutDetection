import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# === Load model, feature columns, and label encoders ===
with open('trained_model.sav', 'rb') as f:
    loaded_model, feature_columns, label_encoders = pickle.load(f)

# === Mapping of friendly questions to actual feature names ===
question_map = {
    "What is your Gender?": "Gender",
    "What is your Occupation?": "Occupation",
    "Are you self-employed?": "self_employed",
    "Does any of your family have mental health problems?": "family_history",
    "Have you ever received mental health treatment?": "treatment",
    "How many days do you spend indoors (never go out)?": "Days_indoors",
    "Do your habits frequently change?": "changing_habits",
    "Does your mood always swing?": "mood_swings",
    "Are you coping well with struggles?": "coping_struggles",
    "Do you still have interest in work?": "interest_to_work",
    "Do you have difficulty socializing with others?": "difficulty_socializing",
    "Have you ever attended any mental health interview?": "interview",
    "Do you have a mental health therapist available to you?": "availability"
}

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Burnout Stress Detector", page_icon="🧠", layout="centered")

    # Sidebar
    with st.sidebar:
        st.title("🧠 Stress Predictor")
        st.markdown("Predict burnout stress level based on user inputs.")
        st.image("https://cdn-icons-png.flaticon.com/512/2282/2282519.png", width=200)
        st.markdown("---")
        st.caption("Developed by Atirah 💻")

    # Main Header
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>Burnout Stress Level Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: gray;'>Answer the following questions to predict your stress level 📊</h5>", unsafe_allow_html=True)
    st.markdown("---")

    # Collect User Input
    st.subheader("📝 Survey Questions")

    user_input = {}
    for question, feature in question_map.items():
        col1, col2 = st.columns([1, 2])
        with col2:
            if feature in label_encoders:
                options = list(label_encoders[feature].classes_)
                user_input[feature] = st.selectbox(f"{question}", options, key=feature)
            else:
                user_input[feature] = st.number_input(f"{question}", step=1.0, key=feature)

    # Prediction
    st.markdown("### 🎯 Prediction")
    if st.button("🚀 Predict Now"):
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
            st.success(f"🌟 **Predicted Growing Stress Level:** `{label}`")
        else:
            st.success(f"🌟 **Predicted Growing Stress Level (encoded):** `{pred}`")

        st.balloons()

if __name__ == '__main__':
    main()
