"""
Script to train and save the maternal health risk prediction model
Run this script after completing your Jupyter notebook analysis
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

def train_and_save_model():
    """Train the model and save it for the Streamlit app"""
    
    print("Fetching dataset...")
    # Fetch dataset
    try:
        maternal_health_risk = fetch_ucirepo(id=863)
        X = maternal_health_risk.data.features
        y = maternal_health_risk.data.targets
    except Exception as e:
        maternal_health_risk = pd.read_csv('data/maternal_health_set.csv')
        X = maternal_health_risk.drop("RiskLevel", axis=1)
        y = maternal_health_risk['RiskLevel']
    
    print("Engineering features...")
    # Feature engineering
    X_engineered = X.copy()
    X_engineered['PulsePressure'] = X_engineered['SystolicBP'] - X_engineered['DiastolicBP']
    X_engineered['MAP'] = X_engineered['DiastolicBP'] + (X_engineered['PulsePressure'] / 3)
    
    print("Encoding labels...")
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Splitting data...")
    # Split data
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X_engineered, y_encoded, test_size=0.2, random_state=1
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=0.25, random_state=1
    )
    
    # Select all features
    selected_features = list(X_engineered.columns)
    
    print("Training XGBoost model...")
    # Train final model
    final_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    final_model.fit(X_train[selected_features], y_train)
    
    # Evaluate on validation set
    y_pred_val = final_model.predict(X_val[selected_features])
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Evaluate on test set
    y_pred_test = final_model.predict(X_test[selected_features])
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nSaving model...")
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("✓ Model saved as 'model.pkl'")
    
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("✓ Label encoder saved as 'label_encoder.pkl'")
    
    print("\nModel training and saving completed!")
    print(f"Risk levels: {le.classes_}")
    print(f"Features used: {selected_features}")

if __name__ == "__main__":
    train_and_save_model()

