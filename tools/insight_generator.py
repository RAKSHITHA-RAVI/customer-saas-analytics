"""
insight_generator.py
--------------------
Uses Claude API to interpret customer segments and generate:
- A descriptive name per segment
- A customer profile narrative
- Key risks and opportunities
- Specific recommended actions
"""

import json
import anthropic
import pandas as pd

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-20250514"


def _format_profiles_for_prompt(profiles: dict, clustered_df: pd.DataFrame) -> str:
    """Build a structured text summary of each segment for the prompt."""
    lines = []
    for seg_id, profile in profiles.items():
        seg_df  = clustered_df[clustered_df["segment"] == seg_id]
        stats   = profile["stats"]
        lines.append(f"""
SEGMENT {seg_id}  ({profile['account_count']} accounts)
─────────────────────────────────────────
Churn rate:            {profile['churn_rate_pct']}%
Avg MRR:               ${stats.get('avg_mrr', 0):,.0f}
Avg ARR:               ${stats.get('total_arr', 0):,.0f}
Avg seats:             {stats.get('seats', 0):.1f}
Unique features used:  {stats.get('unique_features_used', 0):.1f}
Usage duration (hrs):  {stats.get('total_usage_duration_hrs', 0):.1f}
Avg support tickets:   {stats.get('total_tickets', 0):.1f}
Avg satisfaction:      {stats.get('avg_satisfaction', 0):.2f} / 5
Escalation count:      {stats.get('escalation_count', 0):.1f}
Churn events:          {stats.get('churn_event_count', 0):.1f}
Total refunds:         ${stats.get('total_refund_usd', 0):,.0f}
Auto-renew rate:       {stats.get('pct_auto_renew', 0):.1f}%
Upgraded before:       {stats.get('has_upgraded', 0):.0%}
Downgraded before:     {stats.get('has_downgraded', 0):.0%}
Top industries:        {profile['top_industries']}
Top plan tiers:        {profile['top_plans']}
Top churn reasons:     {seg_df['top_churn_reason'].value_counts().head(3).to_dict()}
""")
    return "\n".join(lines)


def generate_segment_insights(
    profiles: dict,
    clustered_df: pd.DataFrame,
    user_query: str = "Give me a full analysis of my customer segments."
) -> dict:
    """
    Call Claude API to analyze segments and return structured insights.
    Returns a dict with per-segment names, narratives, risks, and actions.
    """
    segment_summary = _format_profiles_for_prompt(profiles, clustered_df)

    prompt = f"""
You are a senior SaaS customer success analyst working with B2B subscription data.

USER QUESTION: {user_query}

Below are {len(profiles)} customer segments derived from clustering 500 SaaS accounts
across subscription, feature usage, support, and churn data.

{segment_summary}

For EACH segment, respond in the following JSON format exactly:

{{
  "segments": [
    {{
      "segment_id": 0,
      "name": "A short, memorable 2-4 word label (e.g. 'High-Value Champions')",
      "tagline": "One sentence description of this customer type",
      "profile": "2-3 sentences describing who these customers are, their behavior, and what makes them distinct",
      "risks": ["risk 1", "risk 2", "risk 3"],
      "opportunities": ["opportunity 1", "opportunity 2"],
      "recommended_actions": [
        {{
          "action": "Specific action title",
          "detail": "1-2 sentence explanation of what to do and why"
        }},
        {{
          "action": "Specific action title",
          "detail": "1-2 sentence explanation of what to do and why"
        }},
        {{
          "action": "Specific action title",
          "detail": "1-2 sentence explanation of what to do and why"
        }}
      ],
      "health_score": <integer 1-10 representing overall account health>
    }}
  ],
  "executive_summary": "3-4 sentences summarizing the overall state of the customer base, the biggest opportunity, and the biggest risk",
  "top_priority_action": "The single most important thing the team should do this quarter"
}}

Return ONLY valid JSON. No markdown, no preamble, no explanation outside the JSON.
"""

    print("🤖 Calling Claude API for segment insights...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    insights = json.loads(raw)
    print("✅ Insights generated successfully\n")
    return insights


def print_insights(insights: dict):
    """Pretty-print insights to terminal."""
    print("=" * 60)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 60)
    print(insights.get("executive_summary", ""))
    print(f"\n🎯 TOP PRIORITY: {insights.get('top_priority_action', '')}")

    print("\n" + "=" * 60)
    print("🧩 SEGMENT BREAKDOWN")
    print("=" * 60)

    for seg in insights.get("segments", []):
        print(f"\n▶ Segment {seg['segment_id']}: {seg['name']}  [Health: {seg['health_score']}/10]")
        print(f"  {seg['tagline']}")
        print(f"\n  Profile:\n  {seg['profile']}")

        print("\n  ⚠️  Risks:")
        for r in seg.get("risks", []):
            print(f"     • {r}")

        print("\n  💡 Opportunities:")
        for o in seg.get("opportunities", []):
            print(f"     • {o}")

        print("\n  ✅ Recommended Actions:")
        for a in seg.get("recommended_actions", []):
            print(f"     [{a['action']}]")
            print(f"      → {a['detail']}")

        print()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_master_df
    from cleaner import clean
    from clustering import run_clustering

    master = build_master_df()
    clean_df, scaled_df, scaler = clean(master)
    clustered_df, profiles = run_clustering(clean_df, scaled_df)

    insights = generate_segment_insights(
        profiles,
        clustered_df,
        user_query="Who are my customer segments and what should I do about each?"
    )
    print_insights(insights)
    