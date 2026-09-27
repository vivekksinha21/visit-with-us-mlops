import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️", layout="centered")

MODEL_PATH = Path(__file__).parent / "best_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("✈️ Visit with Us – Wellness Package Predictor")
st.write("Will this customer buy the new Wellness Tourism Package?")

model = load_model()

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration = st.slider("Duration of Pitch (min)", 5, 40, 15)
        num_persons = st.slider("Number of Persons Visiting", 1, 5, 2)
        num_followups = st.slider("Number of Follow-ups", 1, 6, 3)
        preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        num_trips = st.slider("Number of Trips (yearly)", 1, 10, 2)
        num_children = st.slider("Children Visiting (<5 yrs)", 0, 3, 0)

    with col2:
        typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        product = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
        own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=22000, step=500)

    submitted = st.form_submit_button("Predict")

if submitted:
    # hard-coded mappings matching LabelEncoder order after Gender fix
    contact_map = {"Company Invited": 0, "Self Enquiry": 1}
    occ_map     = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
    gender_map  = {"Female": 0, "Male": 1}
    product_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
    marital_map = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
    desig_map   = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

    row = {
        "Age": age,
        "TypeofContact": contact_map[typeof_contact],
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occ_map[occupation],
        "Gender": gender_map[gender],
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_map[product],
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_map[marital],
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": desig_map[designation],
        "MonthlyIncome": monthly_income,
    }
    X = pd.DataFrame([row])

    proba = model.predict_proba(X)[0, 1]
    pred  = int(proba >= 0.5)

    st.subheader("Result")
    if pred == 1:
        st.success(f"Likely to purchase  (probability = {proba:.1%})")
    else:
        st.info(f"Unlikely to purchase  (probability = {proba:.1%})")

    st.progress(float(proba))
    st.caption("Model: tree-based classifier trained on historical campaign data.")
