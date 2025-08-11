import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import random

# Load your trained model
model = joblib.load('xgboost_grid_search.joblib')  # Change to your model path

features = ['BMI', 'Age', 'wc', 'Hc', 'Alcohol', 'FBS']

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966483.png", width=100)
st.sidebar.title("About")
st.sidebar.markdown("""
**Diabetes Risk Predictor**  
Enter patient data to classify as Diabetic or Prediabetic.  
Model: XGBoost  
Features: BMI, Age, Waist, Hip, Alcohol, FBS
""")

# Diabetes tips or fun facts in sidebar
diabetes_tips = [
    "💧 Drink plenty of water to help control blood sugar levels.",
    "🥗 Eat a balanced diet rich in fiber and low in processed sugars.",
    "🚶‍♂️ Regular physical activity helps improve insulin sensitivity.",
    "🩺 Monitor your blood sugar regularly for better management.",
    "😴 Good sleep is important for diabetes control.",
    "🍎 Choose whole fruits over fruit juices for more fiber.",
    "🧘‍♀️ Manage stress with relaxation techniques like yoga or meditation."
]
if st.sidebar.button("Show me a diabetes tip!"):
    tip = random.choice(diabetes_tips)
    st.sidebar.success(tip)
else:
    st.sidebar.info("Click the button for a diabetes tip!")

# Main layout
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🧬 Diabetes vs Prediabetes Predictor</h1>",
    unsafe_allow_html=True
)
st.write("Fill in the details below:")

# Input form with columns
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        bmi = st.number_input("BMI (Body Mass Index):", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
        age = st.number_input("Age:", min_value=1, max_value=120, value=40, step=1)
    with col2:
        wc = st.number_input("Waist Circumference (cm):", min_value=30.0, max_value=200.0, value=90.0, step=0.5)
        hc = st.number_input("Hip Circumference (cm):", min_value=30.0, max_value=200.0, value=95.0, step=0.5)
    with col3:
        fbs = st.number_input("FBS (Fasting Blood Sugar):", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        alcohol_input = st.selectbox("Alcohol Consumption:", ['Yes', 'No', 'Unknown'])
        alcohol = 1 if alcohol_input == 'Yes' else (0 if alcohol_input == 'No' else -1)

    submit = st.form_submit_button("🔎 Predict")

if submit:
    user_input = {
        'BMI': bmi,
        'Age': age,
        'wc': wc,
        'Hc': hc,
        'Alcohol': alcohol,
        'FBS': fbs
    }
    try:
        input_df = pd.DataFrame([user_input])[features]
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.markdown("---")
        st.subheader("🔍 Prediction Result:")
        if prediction[0] == 1:
            st.error("⚠️ The model predicts **Diabetic**.")
        else:
            st.success("✅ The model predicts **Prediabetic**.")

        # Ensure probabilities are between 0 and 1
        st.progress(float(prediction_proba[0][1]))
        st.write("Diabetic Probability")
        st.progress(float(prediction_proba[0][0]))
        st.write("Prediabetic Probability")

        st.write(f"**Probability of Prediabetic:** `{prediction_proba[0][0]:.2f}`")
        st.write(f"**Probability of Diabetic:** `{prediction_proba[0][1]:.2f}`")

        # Feature Importance Bar Chart
        if hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots()
            ax.barh(features, model.feature_importances_, color="#2E86C1")
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        # Pie Chart for Prediction Probability
        fig2, ax2 = plt.subplots()
        labels = ['Prediabetic', 'Diabetic']
        sizes = [float(prediction_proba[0][0]), float(prediction_proba[0][1])]
        colors = ['#58D68D', '#E74C3C']
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.axis('equal')
        st.pyplot(fig2)

        # Display Input Data as Table
        st.markdown("#### Your Input Data")
        st.table(input_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.info("Model is based on XGBoost classification using 6 features.")

## ...existing code...
st.markdown("""
    <style>
        body {
            background-color: #F4F6F7;
        }
        .stProgress > div > div > div > div {
            background-color: #2E86C1;
        }
    </style>
    """, unsafe_allow_html=True)
# ...existing code...