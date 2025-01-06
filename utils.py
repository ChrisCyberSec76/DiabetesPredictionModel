# utils.py
# This script contains utility functions for data loading, preprocessing, and visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

def load_data(data_path):
    """
    Load and preprocess the diabetes dataset
    Returns features (X) and target (y), and feature names
    """
    # Load the diabetes dataset
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target > diabetes.target.mean()  # Convert to binary classification

    # Store feature names
    feature_names = list(X.columns)

    return X, y, feature_names

def plot_feature_importance(model, feature_names):
    """
    Create and save a feature importance plot
    """
    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame for plotting
    feat_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_importance, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()