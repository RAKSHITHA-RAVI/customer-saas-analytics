"""
clustering.py
-------------
Clusters customers into segments using KMeans.
Automatically finds the optimal number of clusters (k)
using the elbow method, then assigns and profiles each segment.
"""

import pandas as pd
import numpy as np
import sklearn.cluster
import sklearn.metrics


def find_optimal_k(scaled_df: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> int:
    """
    Use silhouette score to find the best k.
    Higher silhouette = better-separated clusters.
    """
    best_k = k_min
    best_score = -1

    print(f"🔍 Finding optimal clusters (k={k_min} to k={k_max})...")
    for k in range(k_min, k_max + 1):
        km = sklearn.cluster.KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_df)
        score = sklearn.metrics.silhouette_score(scaled_df, labels)
        print(f"   k={k} → silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✅ Optimal k = {best_k} (score: {best_score:.4f})")
    return best_k


def run_clustering(
    clean_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    n_clusters: int = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run KMeans clustering.
    Returns:
        clustered_df  - clean_df with 'segment' column added
        profiles      - dict of per-segment summary stats
    """
    if n_clusters is None:
        n_clusters = find_optimal_k(scaled_df)

    print(f"\n🤖 Running KMeans with k={n_clusters}...")
    km = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clean_df = clean_df.copy()
    clean_df["segment"] = km.fit_predict(scaled_df)

    # Sort segments by avg MRR descending so segment 0 = highest value
    seg_mrr = clean_df.groupby("segment")["avg_mrr"].mean()
    rank_map = {old: new for new, old in enumerate(seg_mrr.sort_values(ascending=False).index)}
    clean_df["segment"] = clean_df["segment"].map(rank_map)

    print(f"✅ Assigned {n_clusters} segments to {len(clean_df)} accounts\n")

    # Build segment profiles
    profiles = build_segment_profiles(clean_df)

    return clean_df, profiles


def build_segment_profiles(df: pd.DataFrame) -> dict:
    """Build a summary profile for each segment."""
    profiles = {}
    PROFILE_COLS = [
        "avg_mrr", "total_arr", "seats",
        "unique_features_used", "total_usage_duration_hrs",
        "total_tickets", "avg_satisfaction", "escalation_count",
        "churn_event_count", "total_refund_usd",
        "pct_auto_renew", "has_upgraded", "has_downgraded",
    ]

    for seg_id in sorted(df["segment"].unique()):
        seg = df[df["segment"] == seg_id]
        available = [c for c in PROFILE_COLS if c in seg.columns]

        profile = {
            "segment_id":    seg_id,
            "account_count": len(seg),
            "churn_rate_pct": round(seg["churn_flag"].mean() * 100, 1),
            "top_industries": seg["industry"].value_counts().head(2).to_dict(),
            "top_plans":      seg["plan_tier"].value_counts().head(2).to_dict(),
            "stats":          seg[available].mean().round(2).to_dict(),
        }
        profiles[seg_id] = profile

        print(f"── Segment {seg_id} ({len(seg)} accounts) ──")
        print(f"   Churn rate:       {profile['churn_rate_pct']}%")
        print(f"   Avg MRR:          ${profile['stats'].get('avg_mrr', 0):,.0f}")
        print(f"   Avg features used:{profile['stats'].get('unique_features_used', 0):.1f}")
        print(f"   Avg satisfaction: {profile['stats'].get('avg_satisfaction', 0):.2f}")
        print(f"   Top industries:   {profile['top_industries']}")
        print(f"   Top plans:        {profile['top_plans']}")
        print()

    return profiles


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_master_df
    from cleaner import clean

    master = build_master_df()
    clean_df, scaled_df, scaler = clean(master)
    clustered_df, profiles = run_clustering(clean_df, scaled_df)

    print(f"📊 Segment distribution:")
    print(clustered_df["segment"].value_counts().sort_index().to_string())