import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Maternal Health Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Maternal Health Risk Prediction")
st.markdown("""
This application predicts maternal health risk levels based on various health indicators.
The model uses XGBoost classifier trained on maternal health data.
""")

# Load the model
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path("model.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# Load label encoder
@st.cache_resource
def load_label_encoder():
    """Load the label encoder"""
    le_path = Path("label_encoder.pkl")
    if le_path.exists():
        with open(le_path, 'rb') as f:
            return pickle.load(f)
    return None

def calculate_engineered_features(systolic_bp, diastolic_bp):
    """Calculate engineered features"""
    pulse_pressure = systolic_bp - diastolic_bp
    map_value = diastolic_bp + (pulse_pressure / 3)
    return pulse_pressure, map_value

def predict_risk(model, label_encoder, features):
    """Make prediction and return risk level"""
    # Create feature array in the correct order
    feature_array = np.array([[
        features['Age'],
        features['SystolicBP'],
        features['DiastolicBP'],
        features['BS'],
        features['BodyTemp'],
        features['HeartRate'],
        features['PulsePressure'],
        features['MAP']
    ]])
    
    # Make prediction
    prediction = model.predict(feature_array)
    prediction_proba = model.predict_proba(feature_array)
    
    # Decode prediction
    risk_level = label_encoder.inverse_transform(prediction)[0]
    
    return risk_level, prediction_proba[0]

# Main application
def main():
    model = load_model()
    label_encoder = load_label_encoder()
    
    if model is None:
        st.error("⚠️ Model file not found. Please ensure 'model.pkl' is in the same directory.")
        st.info("To train and save the model, run your Jupyter notebook and save the model using pickle.")
        return
    
    if label_encoder is None:
        st.error("⚠️ Label encoder file not found. Please ensure 'label_encoder.pkl' is in the same directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        
        age = st.number_input(
            "Age (years)",
            min_value=10,
            max_value=100,
            value=25,
            help="Patient's age in years"
        )
        
        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=70,
            max_value=200,
            value=120,
            help="Upper number in blood pressure reading"
        )
        
        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=130,
            value=80,
            help="Lower number in blood pressure reading"
        )
        
        blood_sugar = st.number_input(
            "Blood Sugar (mmol/L)",
            min_value=5.0,
            max_value=20.0,
            value=7.5,
            step=0.1,
            help="Blood glucose level"
        )
    
    with col2:
        st.subheader("Vital Signs")
        
        body_temp = st.number_input(
            "Body Temperature (°F)",
            min_value=95.0,
            max_value=105.0,
            value=98.6,
            step=0.1,
            help="Body temperature in Fahrenheit"
        )
        
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40,
            max_value=140,
            value=70,
            help="Heart rate in beats per minute"
        )
        
        # Calculate and display engineered features
        pulse_pressure, map_value = calculate_engineered_features(systolic_bp, diastolic_bp)
        
        st.info(f"**Calculated Values:**\n\n"
                f"Pulse Pressure: {pulse_pressure:.1f} mmHg\n\n"
                f"Mean Arterial Pressure: {map_value:.1f} mmHg")
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Predict Risk Level", type="primary", use_container_width=True):
        # Prepare features
        features = {
            'Age': age,
            'SystolicBP': systolic_bp,
            'DiastolicBP': diastolic_bp,
            'BS': blood_sugar,
            'BodyTemp': body_temp,
            'HeartRate': heart_rate,
            'PulsePressure': pulse_pressure,
            'MAP': map_value
        }
        
        # Make prediction
        risk_level, probabilities = predict_risk(model, label_encoder, features)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Color-code risk level
        if risk_level == "high risk":
            st.error(f"### 🔴 Risk Level: {risk_level.upper()}")
        elif risk_level == "mid risk":
            st.warning(f"### 🟡 Risk Level: {risk_level.upper()}")
        else:
            st.success(f"### 🟢 Risk Level: {risk_level.upper()}")
        
        # Display probabilities
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Risk Level': label_encoder.classes_,
            'Probability': [f"{p*100:.2f}%" for p in probabilities]
        })
        
        col1, col2, col3 = st.columns(3)
        for idx, (level, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
            with [col1, col2, col3][idx]:
                st.metric(level.title(), f"{prob*100:.1f}%")
        
        # Display input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(
                pd.DataFrame([features]).T.rename(columns={0: 'Value'}),
                use_container_width=True
            )
    
    # Add information section
    st.markdown("---")
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Model Information
        - **Algorithm**: XGBoost Classifier
        - **Features Used**: 8 features including original health indicators and engineered features
        - **Risk Categories**: Low Risk, Mid Risk, High Risk
        
        ### Feature Descriptions
        - **Age**: Patient's age in years
        - **Systolic BP**: Upper blood pressure reading
        - **Diastolic BP**: Lower blood pressure reading
        - **Blood Sugar**: Blood glucose level
        - **Body Temperature**: Body temperature in Fahrenheit
        - **Heart Rate**: Heart rate in beats per minute
        - **Pulse Pressure**: Difference between systolic and diastolic BP (calculated)
        - **MAP**: Mean Arterial Pressure (calculated)
        
        ### Important Note
        This tool is for educational purposes only and should not replace professional medical advice.
        Always consult with healthcare professionals for medical decisions.
        """)

if __name__ == "__main__":
    main()