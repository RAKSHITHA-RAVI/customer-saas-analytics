"""
cleaner.py
----------
Cleans and preprocesses the master dataframe for clustering.
- Handles missing values
- Encodes categorical columns
- Scales numeric features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Columns used for clustering (numeric + encodable)
CLUSTER_FEATURES = [
    "avg_mrr",
    "total_arr",
    "seats",
    "total_subscriptions",
    "pct_auto_renew",
    "has_upgraded",
    "has_downgraded",
    "total_usage_events",
    "unique_features_used",
    "total_usage_duration_hrs",
    "avg_error_rate",
    "beta_feature_usage",
    "total_tickets",
    "avg_resolution_hrs",
    "avg_satisfaction",
    "escalation_count",
    "urgent_ticket_count",
    "churn_event_count",
    "total_refund_usd",
    "plan_tier_encoded",
]

CATEGORICAL_COLS = ["plan_tier", "industry", "country", "referral_source"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and add encoded versions."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN values sensibly."""
    df = df.copy()

    # Numeric: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categorical: fill with 'unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale cluster features using StandardScaler. Returns scaled df + scaler."""
    scaler = StandardScaler()
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    scaled_vals = scaler.fit_transform(df[available])
    scaled_df = pd.DataFrame(scaled_vals, columns=available, index=df.index)
    return scaled_df, scaler


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Full cleaning pipeline.
    Returns:
        clean_df    - human-readable cleaned dataframe
        scaled_df   - scaled features ready for clustering
        scaler      - fitted scaler (for inverse transforms later)
    """
    print("🧹 Starting data cleaning...")

    df = encode_categoricals(df)
    print(f"   ✅ Encoded categoricals: {CATEGORICAL_COLS}")

    df = fill_missing(df)
    missing = df.isnull().sum().sum()
    print(f"   ✅ Missing values remaining: {missing}")

    scaled_df, scaler = scale_features(df)
    print(f"   ✅ Scaled {len(scaled_df.columns)} features for clustering")
    print(f"   ✅ Clean dataframe: {df.shape[0]} rows × {df.shape[1]} cols")

    return df, scaled_df, scaler


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_master_df

    master = build_master_df()
    clean_df, scaled_df, scaler = clean(master)

    print("\n📋 Cluster features used:")
    for f in scaled_df.columns:
        print(f"   - {f}")

    print(f"\n🔍 Sample scaled row (first account):")
    print(scaled_df.iloc[0].to_string())