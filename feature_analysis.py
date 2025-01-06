# feature_analysis.py
# Script for detailed feature analysis and relationship visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from utils import load_data
from config import RESULTS_PATH

def analyze_features():
    """
    Perform detailed feature analysis and create visualizations
    """
    # Load data
    print("Loading and preparing data...")
    X, y, feature_names = load_data(None)

    # Create a DataFrame with features and target
    df = X.copy()
    df['target'] = y

    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'correlation_heatmap.png')
    plt.close()

    # Create pairplot for top features
    top_features = ['s5', 'bmi', 'bp', 's3']  # Based on feature importance
    plt.figure(figsize=(12, 8))
    sns.pairplot(df[top_features + ['target']], hue='target')
    plt.savefig(RESULTS_PATH / 'top_features_pairplot.png')
    plt.close()

    # Analyze model performance with different feature combinations
    analyze_feature_combinations(X, y, feature_names)

    # Create ROC curve for different feature sets
    create_roc_curves(X, y, feature_names)

def analyze_feature_combinations(X, y, feature_names):
    """
    Test different feature combinations and their impact on model performance
    """
    # Test different feature combinations
    feature_sets = {
        'all_features': feature_names,
        'top_2': ['s5', 'bmi'],
        'top_4': ['s5', 'bmi', 'bp', 's3'],
        'top_6': ['s5', 'bmi', 'bp', 's3', 's4', 's6']
    }

    results = []
    scaler = StandardScaler()

    for name, features in feature_sets.items():
        # Prepare data
        X_subset = X[features]
        X_scaled = scaler.fit_transform(X_subset)

        # Train model and get cross-validation scores
        model = RandomForestClassifier(random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        results.append({
            'Feature Set': name,
            'Mean CV Score': scores.mean(),
            'Std CV Score': scores.std()
        })

    # Print results
    print("\nPerformance with different feature combinations:")
    for result in results:
        print(f"\n{result['Feature Set']}:")
        print(f"Mean CV Score: {result['Mean CV Score']:.4f} ± {result['Std CV Score']:.4f}")

def create_roc_curves(X, y, feature_names):
    """
    Create ROC curves for different feature combinations
    """
    feature_sets = {
        'All Features': feature_names,
        'Top 2': ['s5', 'bmi'],
        'Top 4': ['s5', 'bmi', 'bp', 's3'],
        'Top 6': ['s5', 'bmi', 'bp', 's3', 's4', 's6']
    }

    plt.figure(figsize=(10, 8))
    scaler = StandardScaler()

    for name, features in feature_sets.items():
        # Prepare data
        X_subset = X[features]
        X_scaled = scaler.fit_transform(X_subset)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model and get predictions
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Feature Sets')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_PATH / 'roc_curves.png')
    plt.close()

if __name__ == "__main__":
    print("Starting feature analysis...")
    analyze_features()
    print("\nAnalysis complete. Check the results directory for visualizations.")