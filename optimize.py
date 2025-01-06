# optimize.py
# Script for model optimization using cross-validation and hyperparameter tuning

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
from config import MODEL_PATH, RESULTS_PATH

def optimize_model():
    """
    Perform model optimization using GridSearchCV and create performance visualizations
    """
    # Load data
    print("Loading data...")
    X, y, feature_names = load_data(None)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize model
    rf = RandomForestClassifier(random_state=42)

    # Perform GridSearchCV
    print("Performing grid search...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_scaled, y)

    # Print best parameters and score
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Create learning curves
    create_learning_curves(grid_search.best_estimator_, X_scaled, y)

    # Feature importance analysis
    analyze_feature_importance(grid_search.best_estimator_, feature_names)

    return grid_search.best_params_

def create_learning_curves(model, X, y):
    """
    Create and save learning curves plot
    """
    # Calculate cross-validation scores for different training set sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5, n_jobs=-1
    )

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, label='Training score')
    plt.plot(train_sizes_abs, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes_abs, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes_abs, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(RESULTS_PATH / 'learning_curves.png')
    plt.close()

def analyze_feature_importance(model, feature_names):
    """
    Create detailed feature importance analysis
    """
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    # Print feature ranking
    print("\nFeature ranking:")
    for f in range(len(feature_names)):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]],
                               importance[indices[f]]))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importance[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices],
               rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'feature_importance_detailed.png')
    plt.close()

if __name__ == "__main__":
    from sklearn.model_selection import learning_curve
    print("Starting model optimization...")
    best_params = optimize_model()
    print("\nOptimization complete. Check the results directory for visualizations.")