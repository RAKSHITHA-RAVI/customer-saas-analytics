"""
data_loader.py
--------------
Loads and merges all 5 RavenStack CSVs into a single
master customer-level dataframe for the agent to work with.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load all 5 CSVs as raw dataframes."""
    tables = {
        "accounts":        pd.read_csv(f"{DATA_DIR}/ravenstack_accounts.csv"),
        "subscriptions":   pd.read_csv(f"{DATA_DIR}/ravenstack_subscriptions.csv"),
        "feature_usage":   pd.read_csv(f"{DATA_DIR}/ravenstack_feature_usage.csv"),
        "support_tickets": pd.read_csv(f"{DATA_DIR}/ravenstack_support_tickets.csv"),
        "churn_events":    pd.read_csv(f"{DATA_DIR}/ravenstack_churn_events.csv"),
    }
    print(f"✅ Loaded raw tables:")
    for name, df in tables.items():
        print(f"   {name}: {df.shape[0]} rows × {df.shape[1]} cols")
    return tables


def build_subscription_features(subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subscription data per account."""
    agg = subscriptions.groupby("account_id").agg(
        total_subscriptions   = ("subscription_id", "count"),
        avg_mrr               = ("mrr_amount", "mean"),
        total_arr             = ("arr_amount", "sum"),
        has_upgraded          = ("upgrade_flag", "max"),
        has_downgraded        = ("downgrade_flag", "max"),
        sub_churn_flag        = ("churn_flag", "max"),
        pct_auto_renew        = ("auto_renew_flag", "mean"),
        pct_trial             = ("is_trial", "mean"),
    ).reset_index()
    agg["avg_mrr"]      = agg["avg_mrr"].round(2)
    agg["total_arr"]    = agg["total_arr"].round(2)
    agg["pct_auto_renew"] = (agg["pct_auto_renew"] * 100).round(1)
    agg["pct_trial"]    = (agg["pct_trial"] * 100).round(1)
    return agg


def build_feature_usage_features(
    feature_usage: pd.DataFrame,
    subscriptions: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate feature usage per account via subscription_id → account_id."""
    merged = feature_usage.merge(
        subscriptions[["subscription_id", "account_id"]], on="subscription_id", how="left"
    )
    agg = merged.groupby("account_id").agg(
        total_usage_events    = ("usage_id", "count"),
        unique_features_used  = ("feature_name", "nunique"),
        total_usage_duration_hrs = ("usage_duration_secs", lambda x: round(x.sum() / 3600, 2)),
        avg_error_rate        = ("error_count", "mean"),
        beta_feature_usage    = ("is_beta_feature", "sum"),
    ).reset_index()
    agg["avg_error_rate"] = agg["avg_error_rate"].round(3)
    return agg


def build_support_features(support_tickets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate support ticket data per account."""
    agg = support_tickets.groupby("account_id").agg(
        total_tickets             = ("ticket_id", "count"),
        avg_resolution_hrs        = ("resolution_time_hours", "mean"),
        avg_first_response_mins   = ("first_response_time_minutes", "mean"),
        avg_satisfaction          = ("satisfaction_score", "mean"),
        escalation_count          = ("escalation_flag", "sum"),
        urgent_ticket_count       = ("priority", lambda x: (x == "urgent").sum()),
    ).reset_index()
    agg["avg_resolution_hrs"]      = agg["avg_resolution_hrs"].round(2)
    agg["avg_first_response_mins"] = agg["avg_first_response_mins"].round(2)
    agg["avg_satisfaction"]        = agg["avg_satisfaction"].round(2)
    return agg


def build_churn_features(churn_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate churn event data per account."""
    agg = churn_events.groupby("account_id").agg(
        churn_event_count     = ("churn_event_id", "count"),
        total_refund_usd      = ("refund_amount_usd", "sum"),
        is_reactivated        = ("is_reactivation", "max"),
        top_churn_reason      = ("reason_code", lambda x: x.mode()[0] if not x.empty else "unknown"),
    ).reset_index()
    agg["total_refund_usd"] = agg["total_refund_usd"].round(2)
    return agg


def build_master_df() -> pd.DataFrame:
    """
    Master entry point.
    Returns one row per account with all features merged in.
    """
    tables = load_raw_tables()

    accounts      = tables["accounts"]
    subscriptions = tables["subscriptions"]
    feature_usage = tables["feature_usage"]
    support       = tables["support_tickets"]
    churn         = tables["churn_events"]

    # Build feature groups
    sub_feats     = build_subscription_features(subscriptions)
    usage_feats   = build_feature_usage_features(feature_usage, subscriptions)
    support_feats = build_support_features(support)
    churn_feats   = build_churn_features(churn)

    # Merge everything onto accounts
    master = accounts.copy()
    master = master.merge(sub_feats,     on="account_id", how="left")
    master = master.merge(usage_feats,   on="account_id", how="left")
    master = master.merge(support_feats, on="account_id", how="left")
    master = master.merge(churn_feats,   on="account_id", how="left")

    # Fill NaN for accounts with no tickets / usage / churn events
    fill_zeros = [
        "total_tickets", "escalation_count", "urgent_ticket_count",
        "churn_event_count", "total_refund_usd", "is_reactivated",
        "total_usage_events", "unique_features_used",
        "total_usage_duration_hrs", "beta_feature_usage",
    ]
    master[fill_zeros] = master[fill_zeros].fillna(0)

    print(f"\n✅ Master dataframe built: {master.shape[0]} accounts × {master.shape[1]} columns")
    return master


if __name__ == "__main__":
    df = build_master_df()
    print("\n📋 Columns:", list(df.columns))
    print("\n🔍 Sample row:")
    print(df.iloc[0].to_string())
    print("\n📊 Quick stats:")
    print(f"   Industries:   {df['industry'].nunique()} unique")
    print(f"   Plan tiers:   {df['plan_tier'].unique()}")
    print(f"   Churn rate:   {df['churn_flag'].mean()*100:.1f}%")
    print(f"   Avg MRR:      ${df['avg_mrr'].mean():,.0f}")
    print(f"   Avg features: {df['unique_features_used'].mean():.1f}")