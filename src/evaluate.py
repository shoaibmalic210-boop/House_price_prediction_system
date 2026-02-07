"""
House Price Prediction - Model Evaluation Pipeline with Cross-Validation
Evaluates the trained model with comprehensive cross-validation metrics.
"""

from pathlib import Path
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate

from preprocess import preprocess_data

# ===============================
# Configuration
# ===============================
RANDOM_STATE = 42

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "House_price_prediction.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# ===============================
# Evaluation Pipeline
# ===============================
def evaluate():
    """
    Complete evaluation pipeline with cross-validation:
    1. Load and preprocess data
    2. Load trained model
    3. Cross-validation evaluation
    4. Test set evaluation
    5. Detailed metrics and diagnostics
    """
    print("=" * 60)
    print("EVALUATION PIPELINE WITH CROSS-VALIDATION")
    print("=" * 60)

    # Check files exist
    print("\n[1/5] Checking files...")
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
    assert (MODEL_DIR / "random_forest.pkl").exists(), f"Model not found: {MODEL_DIR / 'random_forest.pkl'}"
    print(f"[OK] Dataset found: {DATA_PATH}")
    print(f"[OK] Model found: {MODEL_DIR / 'random_forest.pkl'}")

    # Load data
    print("\n[2/5] Loading and preprocessing data...")
    X, y, _, outlier_info = preprocess_data(DATA_PATH)
    print(f"[OK] Data loaded: {X.shape}")
    print(f"    Original samples: {outlier_info['original_count']}")
    print(f"    Outliers removed: {outlier_info['outliers_removed']} ({outlier_info['outlier_percentage']:.2f}%)")
    print(f"    Final samples: {outlier_info['final_count']}")

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[OK] Train/Test split: {X_train.shape} / {X_test.shape}")

    # Load model
    print("\n[3/5] Loading trained model...")
    model = joblib.load(MODEL_DIR / "random_forest.pkl")
    print("[OK] Model loaded")
    
    # Load tuning results if available
    tuning_results = None
    if (MODEL_DIR / "tuning_results.pkl").exists():
        tuning_results = joblib.load(MODEL_DIR / "tuning_results.pkl")
        print("[OK] Tuning results loaded")

    # Cross-validation evaluation
    print("\n[4/5] Cross-Validation Evaluation (5-Fold)...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Multiple metrics for cross-validation
    scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error', 'neg_mae': 'neg_mean_absolute_error'}
    cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, return_train_score=True)
    
    cv_r2_scores = cv_results['test_r2']
    cv_rmse_scores = np.sqrt(-cv_results['test_neg_mse'])
    cv_mae_scores = -cv_results['test_neg_mae']
    
    print(f"\n  Cross-Validation R² Scores: {cv_r2_scores}")
    print(f"    Mean: {cv_r2_scores.mean():.4f} +/- {cv_r2_scores.std():.4f}")
    print(f"\n  Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"    Mean: {cv_rmse_scores.mean():.4f} +/- {cv_rmse_scores.std():.4f}")
    print(f"\n  Cross-Validation MAE Scores: {cv_mae_scores}")
    print(f"    Mean: {cv_mae_scores.mean():.4f} +/- {cv_mae_scores.std():.4f}")

    # Test set predictions
    print("\n[5/5] Making predictions and calculating metrics...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if tuning_results:
        print("\nHyperparameter Tuning Results:")
        print(f"  Best Parameters: {tuning_results['best_params']}")
        print(f"  Best CV R² Score: {tuning_results['best_cv_score']:.4f}")
    
    print("\nTrain Set Metrics:")
    print(f"  RMSE (log): {train_rmse:.4f}")
    print(f"  MAE  (log): {train_mae:.4f}")
    print(f"  R² Score:  {train_r2:.4f}")

    print("\nTest Set Metrics:")
    print(f"  RMSE (log): {test_rmse:.4f}")
    print(f"  MAE  (log): {test_mae:.4f}")
    print(f"  R² Score:  {test_r2:.4f}")

    print("\nCross-Validation Summary (5-Fold):")
    print(f"  R² - Mean: {cv_r2_scores.mean():.4f}, Std: {cv_r2_scores.std():.4f}")
    print(f"  RMSE - Mean: {cv_rmse_scores.mean():.4f}, Std: {cv_rmse_scores.std():.4f}")
    print(f"  MAE - Mean: {cv_mae_scores.mean():.4f}, Std: {cv_mae_scores.std():.4f}")

    # Overfitting check
    r2_diff = train_r2 - test_r2
    print("\nModel Status:")
    if r2_diff > 0.15:
        print(f"  [WARN] OVERFITTING: Train R² ({train_r2:.4f}) >> Test R² ({test_r2:.4f})")
        print(f"    Recommendation: Increase regularization, reduce model complexity")
    elif r2_diff < -0.1:
        print(f"  [WARN] UNDERFITTING: Test R² ({test_r2:.4f}) >> Train R² ({train_r2:.4f})")
        print(f"    Recommendation: Increase model complexity, reduce regularization")
    else:
        print(f"  [OK] BALANCED: Model generalizes well (diff: {r2_diff:.4f})")
    
    # Cross-validation variance check
    cv_std = cv_r2_scores.std()
    if cv_std > 0.05:
        print(f"  [WARN] HIGH CV VARIANCE: Std Dev = {cv_std:.4f}")
        print(f"    Recommendation: Model performance varies across folds, consider more stable features")
    else:
        print(f"  [OK] STABLE: CV scores show low variance (std: {cv_std:.4f})")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()