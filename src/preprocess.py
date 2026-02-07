import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===============================
# Reproducibility
# ===============================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===============================
# Constants
# ===============================
TARGET = "price"

DROP_COLS = [
    "Unnamed: 0",   # index artifact
    "property_id",  # unique identifier
    "page_url"      # leakage + useless
]

# ===============================
# Utility Functions
# ===============================
def frequency_encoding(X):
    """
    Frequency encoding for high-cardinality categorical variables
    """
    X = pd.DataFrame(X)
    return X.apply(lambda col: col.map(col.value_counts(normalize=True)))


def remove_outliers_iqr(df, target_col, q1=0.25, q3=0.75):
    """
    Remove outliers from target variable using Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column (e.g., 'price')
        q1: Lower quantile (default: 0.25)
        q3: Upper quantile (default: 0.75)
    
    Returns:
        df_clean: DataFrame with outliers removed
        outlier_indices: Indices of removed outliers
    
    Explanation:
        - Q1 (25th percentile) and Q3 (75th percentile) define the middle 50% of data
        - IQR = Q3 - Q1
        - Outliers are values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
        - This is the standard statistical method for outlier detection
    """
    df = df.copy()
    q1_val = df[target_col].quantile(q1)
    q3_val = df[target_col].quantile(q3)
    iqr = q3_val - q1_val
    
    lower_bound = q1_val - 1.5 * iqr
    upper_bound = q3_val + 1.5 * iqr
    
    # Identify outliers
    outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
    outlier_indices = df[outlier_mask].index.tolist()
    
    # Remove outliers
    df_clean = df[~outlier_mask].reset_index(drop=True)
    
    return df_clean, outlier_indices


# ===============================
# Feature Engineering
# ===============================
def create_date_features(df):
    """
    Extract year and month from date_added column
    """
    df = df.copy()
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df.drop(columns=["date_added"], inplace=True)
    return df


# ===============================
# Feature Grouping
# ===============================
def get_feature_groups():
    num_features = ["Total_Area", "latitude(N,S)", "longitude(E,W)"]
    count_features = ["bedrooms", "baths"]
    date_features = ["year_added", "month_added"]

    nominal_features = [
        "property_type",
        "purpose",
        "city",
        "province_name"
    ]

    high_card_features = ["location", "agency", "agent"]
    location_id_features = ["location_id"]

    return (
        num_features,
        count_features,
        date_features,
        nominal_features,
        high_card_features,
        location_id_features
    )


# ===============================
# Pipeline Builder
# ===============================
def build_preprocessor():
    (
        num_features,
        count_features,
        date_features,
        nominal_features,
        high_card_features,
        location_id_features
    ) = get_feature_groups()

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson"))
    ])

    count_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    date_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    high_card_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq", FunctionTransformer(frequency_encoding))
    ])

    location_id_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq", FunctionTransformer(frequency_encoding))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("count", count_pipeline, count_features),
            ("date", date_pipeline, date_features),
            ("nom", nominal_pipeline, nominal_features),
            ("high", high_card_pipeline, high_card_features),
            ("loc_id", location_id_pipeline, location_id_features)
        ],
        remainder="drop"
    )

    return preprocessor


# ===============================
# Main Preprocessing Function
# ===============================
def preprocess_data(csv_path):
    """
    Reads raw CSV, cleans data, removes outliers using IQR method,
    applies preprocessing pipeline, and returns transformed features and target.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        X_clean: Transformed feature matrix (ndarray)
        y: Log-transformed target variable (Series)
        preprocessor: Fitted ColumnTransformer for future predictions
        outlier_info: Dict with outlier removal statistics
    """
    # Load data
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    print(f"Original dataset shape: {df.shape}")

    # Drop unnecessary columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Date feature engineering
    df = create_date_features(df)

    # Remove outliers from target variable using IQR method
    df_clean, outlier_indices = remove_outliers_iqr(df, TARGET)
    
    outlier_info = {
        "original_count": len(df),
        "outliers_removed": len(outlier_indices),
        "final_count": len(df_clean),
        "outlier_percentage": (len(outlier_indices) / len(df)) * 100
    }
    
    print(f"Outliers removed: {outlier_info['outliers_removed']} ({outlier_info['outlier_percentage']:.2f}%)")
    print(f"Clean dataset shape: {df_clean.shape}")

    # Split features and target
    X = df_clean.drop(columns=[TARGET])
    y = np.log1p(df_clean[TARGET])

    # Build and apply preprocessing
    preprocessor = build_preprocessor()
    X_clean = preprocessor.fit_transform(X)

    return X_clean, y, preprocessor, outlier_info



