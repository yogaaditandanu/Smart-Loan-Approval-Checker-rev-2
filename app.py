import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# === Load Model & Tools ===
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- NEW FUNCTION: AUTOMATED CREDIT SCORE CALCULATION ---
def calculate_credit_score_auto(income, percent_income, default, home):
    """
    Estimate credit score based on simplified rules.
    """
    score = 650  # Starting from average

    # 1. Adjustment based on Default History (Most Significant Factor)
    if default == "Yes":
        score -= 150
    else:
        score += 50

    # 2. Adjustment based on Loan-to-Income Ratio
    if percent_income <= 0.20:
        score += 40
    elif percent_income <= 0.40:
        score += 15
    elif percent_income > 0.50:
        score -= 70

    # 3. Adjustment based on Home Ownership
    if home in ["OWN", "MORTGAGE"]:
        score += 30
    else:  # RENT
        score -= 20

    # 4. Adjustment based on Annual Income
    if income > 150_000_000:
        score += 30
    elif income < 30_000_000:
        score -= 25

    # Clamp score between 300 and 850
    score = min(max(score, 300), 850)
    return score
# --- END OF NEW FUNCTION ---


# === Streamlit Page Setup ===
st.set_page_config(page_title="Smart Loan Approval Checker", layout="wide")

# === Sidebar ===
st.sidebar.image("loan.jpg", width=180)
st.sidebar.title("Smart Loan Approval Checker 🔮")
page = st.sidebar.radio("📂 Menu", [
    "🏡 Overview", 
    "📘 User Guide", 
    "🔎 Single Check", 
    "🗃️ Batch Check", 
    "🗣️ Feedback"
])

# === 1. Overview Page ===
if page == "🏡 Overview":
    st.markdown("<h2 style='color:#6C63FF;'>🏦 Smart Loan Approval Checker</h2>", unsafe_allow_html=True)
    st.markdown("""
Welcome to **Smart Loan Approval Checker**! This app helps predict whether a loan application is likely to be **approved** or **rejected**, based on personal and financial data.

---

### 🚀 Key Features
- 🔍 **Single Check** – Instantly check one application  
- 📁 **Batch Check** – Predict multiple applicants via CSV  
- 📘 **User Guide** – Learn about each input  
- 🗣️ **Feedback** – Share your thoughts or suggestions!

---

💡 Perfect for analysts, finance staff, and general users.

🧑‍💻 *Created by Yoga Adi Tandanu*
""")

# === 2. User Guide Page ===
elif page == "📘 User Guide":
    st.markdown("<h2 style='color:#6C63FF;'>📘 How to Use</h2>", unsafe_allow_html=True)

    st.markdown("### 🔍 Single Check")
    st.markdown("""
Fill in the form with the following information:

| Label | Description |
|-------|-------------|
| 🎂 Age | Applicant's age (18–100 years) |
| 🚻 Gender | Select *male* or *female* |
| 🎓 Education | High School, Bachelor, Master, etc. |
| 🏘️ Home Ownership | RENT, MORTGAGE, OWN |
| ❗ Default History | Any prior default? Yes/No |
| 💼 Annual Income | Total yearly income |
| 💳 Loan Amount | Requested loan amount |
| 📊 Interest Rate (%) | Annual loan interest |
| 🧮 Loan-to-Income Ratio | Loan ÷ Income |
| 📉 Credit Score | Ranges from 300 to 850 |
| 🎯 Loan Purpose | VENTURE, MEDICAL, etc. |
""")

    st.markdown("### 🗃️ Batch Check")
    st.markdown("Upload a CSV file with the following columns:")
    st.code("person_age, person_gender, person_education, person_income,\nperson_home_ownership, previous_loan_defaults_on_file,\nloan_amnt, loan_int_rate, loan_percent_income,\ncredit_score, loan_intent")

    st.markdown("### 🗣️ Feedback")
    st.markdown("Use the **Feedback** menu to submit any comments or suggestions.")

# === 3. Single Prediction Page ===
elif page == "🔎 Single Check":
    st.markdown("<h2 style='color:#6C63FF;'>🔎 Single Loan Check</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("#### 📋 Applicant Form")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Age", 18, 100, 30)
            gender = st.selectbox("🚻 Gender", ["male", "female"])
            education = st.selectbox("🎓 Education", ["High School", "Bachelor", "Master", "Associate"])
            home = st.selectbox("🏘️ Home Ownership", ["RENT", "MORTGAGE", "OWN"])
            default = st.selectbox("❗ Previous Loan Default?", ["Yes", "No"])

        with col2:
            income = st.number_input("💼 Annual Income (IDR)", 1000, 1_000_000_000, 50_000_000, step=1_000_000)
            loan_amt = st.number_input("💳 Loan Amount (IDR)", 1000, 100_000_000, 10_000_000, step=1_000_000)
            interest = st.slider("📊 Interest Rate (%)", 5.0, 30.0, 15.0)
            
            ratio_value = round(loan_amt / income, 2) if income > 0 else 0.0
            percent_income = st.slider("🧮 Loan-to-Income Ratio", 0.0, 1.0, ratio_value, disabled=True)
            
            credit_score = calculate_credit_score_auto(income, percent_income, default, home)
            st.slider("📉 Credit Score (Auto)", 300, 850, credit_score, disabled=True)
            
            intent = st.selectbox("🎯 Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION"])

        st.info("💡 Example: Income 50M, Loan 10M → Ratio = 0.2")

        if st.button("🧠 Predict Now"):
            input_df = pd.DataFrame([{
                'person_age': age,
                'person_gender': gender,
                'person_education': education,
                'person_income': income,
                'person_home_ownership': home,
                'loan_amnt': loan_amt,
                'loan_int_rate': interest,
                'loan_percent_income': percent_income,
                'credit_score': credit_score,
                'loan_intent': intent,
                'previous_loan_defaults_on_file': default
            }])

            for col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

            features = model.get_booster().feature_names
            scaled_input = scaler.transform(input_df[features])

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            st.markdown("### 🔮 Prediction Result")
            if prediction == 1:
                st.success("✅ The loan is likely to be **APPROVED**")
            else:
                st.error("❌ The loan is likely to be **REJECTED**")
            st.metric("📈 Approval Probability", f"{probability * 100:.2f} %")

# === 4. Batch Prediction Page ===
elif page == "🗃️ Batch Check":
    st.markdown("<h2 style='color:#6C63FF;'>🗃️ Batch Loan Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])

        features = model.get_booster().feature_names
        df_scaled = scaler.transform(df[features])

        df["prediction"] = model.predict(df_scaled)
        df["approval_prob"] = model.predict_proba(df_scaled)[:, 1]

        st.markdown("### 📋 Prediction Results")
        filter_val = st.slider("🔎 Show data with probability > ", 0.0, 1.0, 0.5)
        st.dataframe(df[df["approval_prob"] > filter_val].head(10))

        st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="loan_prediction_results.csv")

# === 5. Feedback Page ===
elif page == "🗣️ Feedback":
    st.markdown("<h2 style='color:#6C63FF;'>🗣️ Share Your Feedback</h2>", unsafe_allow_html=True)

    name = st.text_input("🧑 Your Name (Optional)")
    rating = st.slider("⭐ Rate This App", 1, 5, 4)
    comments = st.text_area("💬 Suggestions, feedback, or impressions:")

    if st.button("📨 Submit"):
        st.success("Thank you for your feedback!")
        st.markdown(f"**Name**: {name if name else 'Anonymous'}  \n**Rating**: {rating}/5  \n**Comment**: {comments}")
