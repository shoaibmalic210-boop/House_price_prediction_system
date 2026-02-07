"""
House Price Prediction - Model Training Pipeline with Hyperparameter Tuning
Trains a Random Forest model with cross-validation and hyperparameter optimization.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import preprocess_data

# ===============================
# Configuration
# ===============================
RANDOM_STATE = 42

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "House_price_prediction.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter tuning grid (optimized for older systems)
PARAM_GRID = {
    'n_estimators': [30, 50],
    'max_depth': [10, 15],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [3, 5]
}

# ===============================
# Training Pipeline
# ===============================
def train():
    """
    Complete training pipeline with hyperparameter tuning:
    1. Load and preprocess data
    2. Split into train/test
    3. Hyperparameter tuning with GridSearchCV
    4. Cross-validation evaluation
    5. Final evaluation on test set
    6. Save best model
    """
    print("=" * 60)
    print("TRAINING PIPELINE WITH HYPERPARAMETER TUNING")
    print("=" * 60)

    # Load & preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    X, y, preprocessor, outlier_info = preprocess_data(DATA_PATH)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    print(f"[OK] Data loaded: {X.shape}")
    print(f"    Original samples: {outlier_info['original_count']}")
    print(f"    Outliers removed: {outlier_info['outliers_removed']} ({outlier_info['outlier_percentage']:.2f}%)")
    print(f"    Final samples: {outlier_info['final_count']}")

    # Train-test split
    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")

    # Hyperparameter tuning with GridSearchCV
    print("\n[3/5] Hyperparameter Tuning (GridSearchCV with 5-Fold CV)...")
    print(f"  Testing {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_split']) * len(PARAM_GRID['min_samples_leaf'])} parameter combinations...")
    
    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n[OK] Best parameters found: {grid_search.best_params_}")
    print(f"[OK] Best cross-validation R² score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Cross-validation on full training set
    print("\n[4/5] Cross-Validation Evaluation (5-Fold)...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='r2')
    
    print(f"  Fold Scores: {cv_scores}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Final evaluation on test set
    print("\n[5/5] Final Evaluation on Test Set...")
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n  Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    # Save best model
    print("\n[SAVING]")
    joblib.dump(best_model, MODEL_DIR / "random_forest.pkl")
    print(f"[OK] Best model saved to {MODEL_DIR / 'random_forest.pkl'}")
    print(f"[OK] Preprocessor saved to {MODEL_DIR / 'preprocessor.pkl'}")
    
    # Save tuning results
    tuning_results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'cv_scores': cv_scores.tolist(),
        'train_metrics': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
        'test_metrics': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
    }
    joblib.dump(tuning_results, MODEL_DIR / "tuning_results.pkl")
    print(f"[OK] Tuning results saved to {MODEL_DIR / 'tuning_results.pkl'}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return best_model


if __name__ == "__main__":
    train()
