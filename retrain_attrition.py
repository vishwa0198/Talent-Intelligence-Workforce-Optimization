
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define paths
PROCESSED_DIR = r"C:\Guvi\Talent Intelligence & Workforce Optimization\notebooks\processed"
MODELS_DIR = r"C:\Guvi\Talent Intelligence & Workforce Optimization\models"
DATA_FILE = os.path.join(PROCESSED_DIR, "hr_clean.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "attrition_model.pkl")

def train_attrition_model():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Target variable
    if "Attrition" not in df.columns:
        print("Error: 'Attrition' column missing.")
        return

    # Convert target to numeric
    y = df["Attrition"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    
    # Feature selection - identify numeric columns automatically
    # (Simplified for robustness)
    drop_cols = ["Attrition", "EmployeeID", "EmployeeCount", "StandardHours", "Over18"]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # Select only numeric, or encode categorical? 
    # For a quick fix that matches typical "prepare_employee_features", we'll stick to numeric for simplicity
    # OR we can assume the input is already preprocessed?
    # Checking typical HR dataset, it usually has strings. 
    # Let's simple-encode categorical columns to avoid complex pipeline errors.
    
    X = pd.get_dummies(X_raw, drop_first=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features...")
    
    # Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {score:.4f}")
    
    # Save model
    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(pipeline, MODEL_FILE)
    print("Done!")

if __name__ == "__main__":
    train_attrition_model()
