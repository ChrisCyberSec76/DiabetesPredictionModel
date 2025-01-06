# predict.py
# Script for making predictions using the trained model

import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH

def load_model():
    """
    Load the trained model, scaler, and feature names
    """
    model = joblib.load(MODEL_PATH / 'model.pkl')
    scaler = joblib.load(MODEL_PATH / 'scaler.pkl')
    feature_names = joblib.load(MODEL_PATH / 'feature_names.pkl')
    return model, scaler, feature_names

def predict(features):
    """
    Make predictions using the trained model

    Args:
        features (array-like): Input features for prediction

    Returns:
        prediction (int): 0 or 1 indicating the predicted class
        probability (float): Probability of the positive class
    """
    # Load model, scaler, and feature names
    model, scaler, feature_names = load_model()

    # Ensure features are in correct format and have feature names
    if isinstance(features, (list, np.ndarray)):
        features = pd.DataFrame([features], columns=feature_names)

    # Validate input features
    if not all(name in features.columns for name in feature_names):
        raise ValueError(f"Input must contain all required features: {feature_names}")

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return prediction, probability

if __name__ == "__main__":
    # Example of how to use the prediction function
    print("Loading model and making a prediction...")

    # First, let's get the feature names
    _, _, feature_names = load_model()
    print("\nRequired features:", feature_names)

    # Example values for each feature
    sample_features = [0.03807591, 0.05068012, 0.06169621, 0.02187235,
                       -0.0442235, -0.03482076, -0.04340085, -0.00259226,
                       0.01990842, -0.01764613]

    print("\nMaking prediction...")
    pred, prob = predict(sample_features)
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Probability: {prob:.2f}")

    # You can also pass features as a dictionary or DataFrame
    print("\nAlternative way to make prediction using a dictionary:")
    features_dict = dict(zip(feature_names, sample_features))
    pred, prob = predict(pd.DataFrame([features_dict]))
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Probability: {prob:.2f}")