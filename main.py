# main.py
# This script demonstrates a basic supervised learning pipeline using scikit-learn
# We'll create a binary classification model to predict diabetes based on medical measurements

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from utils import load_data, plot_feature_importance
from config import MODEL_PATH, DATA_PATH

def train_model():
    """
    Main training function that orchestrates the ML pipeline
    """
    # Load and preprocess the data
    print("Loading data...")
    X, y, feature_names = load_data(DATA_PATH)

    # Split data into training and testing sets
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)

    # Print model performance
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))

    # Plot feature importance
    plot_feature_importance(model, feature_names)

    # Save the model, scaler, and feature names
    print("\nSaving model, scaler, and feature names...")
    joblib.dump(model, MODEL_PATH / 'model.pkl')
    joblib.dump(scaler, MODEL_PATH / 'scaler.pkl')
    joblib.dump(feature_names, MODEL_PATH / 'feature_names.pkl')

if __name__ == "__main__":
    train_model()