"""
app.py — RavenStack Customer Insights Agent
Light theme · pastel palette · Plotly charts · no API key needed
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from data_loader import build_master_df
from cleaner import clean
from clustering import run_clustering

st.set_page_config(page_title="RavenStack · Insights", page_icon="🤖", layout="wide")

# ─── Design tokens ───────────────────────────────────────────
BG       = "#f8f9fc"
SURFACE  = "#ffffff"
SURFACE2 = "#f1f4f9"
BORDER   = "#e2e8f0"
BORDER2  = "#cbd5e1"
TEXT_H   = "#0f172a"
TEXT_B   = "#334155"
TEXT_DIM = "#94a3b8"

P1     = "#7c9fd4"
P2     = "#a0b8e0"
P_DARK = "#5a80bb"

A_GREEN = "#5aab7a"
A_AMBER = "#e0a550"
A_RED   = "#d97070"

CHART_PAL = ["#7c9fd4","#8db8c8","#9db5e0","#a8c5b8","#b8b0d8","#c8bfa8"]

PLOT_BG  = "#ffffff"
GRID_CLR = "#f1f4f9"
AXIS_CLR = "#94a3b8"
HOVER_BG = "#1e293b"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

.stApp {{ background: {BG}; color: {TEXT_B}; }}
.main .block-container {{ padding: 2rem 2.5rem 3rem; max-width: 1400px; }}

/* ── KPI card ── */
.kpi-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 20px 16px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.kpi-value {{ font-size: 1.75rem; font-weight: 800; line-height: 1.1; margin-bottom: 5px; }}
.kpi-label {{ font-size: 0.63rem; font-weight: 600; color: {TEXT_DIM};
              text-transform: uppercase; letter-spacing: 0.1em; }}

/* ── Summary cards ── */
.summary-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 18px 20px; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.summary-card-label {{
    font-size: 0.62rem; font-weight: 700; color: {TEXT_DIM};
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid {BORDER};
}}
.summary-card-text {{ font-size: 0.87rem; color: {TEXT_B}; line-height: 1.8; }}
.summary-card-text b {{ color: {TEXT_H}; }}

/* ── Segment snapshot rows ── */
.seg-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: 0.85rem;
}}
.seg-row:last-child {{ border-bottom: none; }}
.seg-row-name  {{ font-weight: 600; color: {TEXT_H}; }}
.seg-row-stats {{ display: flex; gap: 14px; color: {TEXT_DIM}; font-size: 0.8rem; }}

/* ── Priority box ── */
.prio-box {{
    background: #fffbeb; border: 1px solid #fcd34d;
    border-left: 4px solid {A_AMBER};
    border-radius: 12px; padding: 14px 20px;
    font-size: 0.85rem; color: #78350f; line-height: 1.7;
}}
.prio-box b {{ color: #92400e; }}

/* ── Segment card ── */
.seg-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.seg-top-bar   {{ height: 3px; border-radius: 3px; margin-bottom: 16px; }}
.seg-name      {{ font-size: 0.92rem; font-weight: 700; color: {TEXT_H}; margin-bottom: 4px; }}
.seg-tag       {{ font-size: 0.72rem; color: {TEXT_DIM}; margin-bottom: 14px; line-height: 1.5; }}
.seg-health-num {{ font-size: 2.2rem; font-weight: 800; line-height: 1; }}
.seg-health-den {{ font-size: 0.78rem; color: {TEXT_DIM}; }}
.seg-health-lbl {{ font-size: 0.6rem; font-weight: 600; color: {TEXT_DIM};
                   text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }}

/* ── Pills ── */
.pill-r {{ display:inline-block; background:#fef2f2; border:1px solid #fecaca;
           color:{A_RED}; border-radius:20px; padding:3px 12px;
           font-size:0.72rem; margin:3px 3px 0 0; font-weight:500; }}
.pill-g {{ display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0;
           color:{A_GREEN}; border-radius:20px; padding:3px 12px;
           font-size:0.72rem; margin:3px 3px 0 0; font-weight:500; }}

/* ── Action items ── */
.action-item {{
    background: {SURFACE2}; border: 1px solid {BORDER};
    border-left: 3px solid {P1}; border-radius: 10px;
    padding: 12px 16px; margin: 8px 0;
}}
.action-title {{ font-size: 0.85rem; font-weight: 700; color: {P_DARK}; margin-bottom: 4px; }}
.action-body  {{ font-size: 0.78rem; color: {TEXT_B}; line-height: 1.6; }}

/* ── Profile box ── */
.profile-box {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 18px; font-size: 0.83rem; color: {TEXT_B}; line-height: 1.75;
}}
.profile-box b {{ color: {TEXT_H}; }}

/* ── Section label ── */
.sec-header {{
    font-size: 0.62rem; font-weight: 700; color: {TEXT_DIM};
    text-transform: uppercase; letter-spacing: 0.14em;
    margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid {BORDER};
}}
.page-title {{ font-size: 2rem; font-weight: 800; color: {TEXT_H}; margin-bottom: 4px; }}
.page-sub   {{ font-size: 0.7rem; font-weight: 600; color: {TEXT_DIM};
               text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 24px; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{ background:{SURFACE} !important; border-right:1px solid {BORDER}; }}
[data-testid="metric-container"]  {{ background:{SURFACE} !important; border:1px solid {BORDER} !important;
                                      border-radius:10px !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]  {{ background:{SURFACE2}; border-radius:10px; padding:3px; }}
.stTabs [data-baseweb="tab"]       {{ color:{TEXT_DIM}; font-size:0.82rem; }}
.stTabs [aria-selected="true"]     {{ background:{SURFACE} !important; color:{TEXT_H} !important;
                                      border-radius:8px !important; box-shadow:0 1px 3px rgba(0,0,0,0.08) !important; }}

hr {{ border-color:{BORDER} !important; margin:1.5rem 0 !important; }}
.stDataFrame {{ border:1px solid {BORDER} !important; border-radius:10px; }}
#MainMenu, footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# ─── Plotly helpers ──────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
    font=dict(family="Inter", color=TEXT_B, size=12),
    margin=dict(l=8, r=8, t=36, b=40),
    hoverlabel=dict(bgcolor=HOVER_BG, font_size=12, font_color="#fff", bordercolor=HOVER_BG),
)
AXIS_S = dict(showgrid=True, gridcolor=GRID_CLR, gridwidth=1,
              showline=True, linecolor=BORDER2, linewidth=1,
              tickfont=dict(size=11, color=AXIS_CLR),
              title_font=dict(size=11, color=TEXT_DIM))
X_AXIS_S = dict(showgrid=False, showline=True, linecolor=BORDER2, linewidth=1,
                tickfont=dict(size=11, color=AXIS_CLR), tickangle=0, automargin=True)


def bar_chart(cats, vals, color=P1, title="", y_title="", height=230):
    short = [c if len(c) <= 18 else c[:16]+"…" for c in cats]
    texts = [f"{v:,.0f}" if isinstance(v,(int,float)) and v>=100
             else f"{v:.1f}" if isinstance(v,float) else str(v) for v in vals]
    fig = go.Figure(go.Bar(
        x=short, y=vals, marker=dict(color=color, line=dict(width=0), opacity=0.9),
        text=texts, textposition="outside",
        textfont=dict(size=11, color=TEXT_B),
        hovertemplate="<b>%{x}</b><br>"+(y_title or "Value")+": <b>%{y}</b><extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=title, font=dict(size=12,color=TEXT_H), x=0, pad=dict(b=6)),
        height=height, xaxis=dict(**X_AXIS_S),
        yaxis=dict(**AXIS_S, title=y_title, range=[0, max(vals)*1.25] if vals else [0,10]),
        bargap=0.35, showlegend=False)
    return fig


def donut_chart(labels, vals, title="", height=230):
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.58,
        marker=dict(colors=CHART_PAL[:len(labels)], line=dict(color="#fff", width=2)),
        textfont=dict(size=11, color="#fff"),
        hovertemplate="<b>%{label}</b>: <b>%{value}</b> (%{percent})<extra></extra>",
        sort=True,
    ))
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=title, font=dict(size=12,color=TEXT_H), x=0),
        height=height, showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM, size=10),
                    orientation="v", x=1, xanchor="left"))
    return fig


def hbar_chart(cats, vals, color=P1, title="", x_title="", height=230):
    fig = go.Figure(go.Bar(
        y=cats, x=vals, orientation="h",
        marker=dict(color=color, line=dict(width=0), opacity=0.9),
        text=[f"{v:,.0f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color=TEXT_B),
        hovertemplate="<b>%{y}</b><br>"+(x_title or "Value")+": <b>%{x}</b><extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(**CHART_LAYOUT,
        title=dict(text=title, font=dict(size=12,color=TEXT_H), x=0),
        height=height, showlegend=False,
        xaxis=dict(**AXIS_S, title=x_title, range=[0, max(vals)*1.3] if vals else [0,10]),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11,color=AXIS_CLR)),
        bargap=0.35)
    return fig


# ─── Helpers ────────────────────────────────────────────────
def hcolor(s): return A_GREEN if s>=7 else (A_AMBER if s>=4 else A_RED)
def hemoji(s): return "🟢" if s>=7 else ("🟡" if s>=4 else "🔴")

def calc_health(churn, sat, esc, tickets):
    score  = 10
    score -= (churn/100)*15
    score -= max(0, (4.5-sat)*1.2)
    score -= esc*0.8
    score -= min(tickets*0.1, 1.5)
    return max(1, min(10, round(score)))


# ─── Insights engine ─────────────────────────────────────────
def derive_insights(profiles, clustered_df):
    seg_mrr = {sid: p["stats"].get("avg_mrr",0) for sid,p in profiles.items()}
    ranked  = sorted(seg_mrr, key=seg_mrr.get, reverse=True)
    labels  = [
        "High-Value Champions",      # rank 1 — highest MRR
        "Engaged but At-Risk",       # rank 2
        "Low-Engagement Drifters",   # rank 3
        "Growth Potential",          # rank 4
        "Early Stage Accounts",      # rank 5
        "At-Risk Tail",              # rank 6 — lowest MRR
    ]
    icons   = ["🏆","⚡","💤","🌱","🔵","⚠️"]
    NAMES   = {sid: f"{icons[i]} {labels[i]}" for i,sid in enumerate(ranked)}

    segments = []
    for seg_id, profile in profiles.items():
        s        = profile["stats"]
        churn    = profile["churn_rate_pct"]
        mrr      = s.get("avg_mrr",0)
        feats    = s.get("unique_features_used",0)
        sat      = s.get("avg_satisfaction",0)
        esc      = s.get("escalation_count",0)
        tickets  = s.get("total_tickets",0)
        duration = s.get("total_usage_duration_hrs",0)
        errors   = s.get("avg_error_rate",0)
        health   = calc_health(churn, sat, esc, tickets)

        seg_df       = clustered_df[clustered_df["segment"]==seg_id]
        top_industry = seg_df["industry"].value_counts().index[0] if len(seg_df) else "N/A"
        top_plan     = seg_df["plan_tier"].value_counts().index[0] if len(seg_df) else "N/A"
        _r = seg_df["top_churn_reason"].replace("unknown",None).dropna().value_counts() \
             if "top_churn_reason" in seg_df.columns else pd.Series(dtype=str)
        top_reason = _r.index[0] if len(_r) else "not recorded"
        name = NAMES.get(seg_id, f"Segment {seg_id}")

        risks = [f"Churn {churn}% — {'above average, urgent' if churn>22 else 'monitor closely'}"]
        if sat < 4.2:    risks.append(f"Satisfaction {sat:.2f}/5 — room to improve")
        if esc > 0.5:    risks.append(f"{esc:.1f} avg escalations — friction present")
        if feats < 25:   risks.append(f"Only {feats:.0f} avg features used — low adoption")
        if errors > 0.8: risks.append(f"Error rate {errors:.2f} — product friction")
        risks.append(f"Top churn reason: {top_reason}")

        opps = ["Prime Enterprise upsell candidates" if mrr>3000 else "Revenue expansion via plan upgrades"]
        opps.append(f"Activate {int(30-feats)} more features to cut churn" if feats<25 else "Power users → case studies")
        if churn < 20: opps.append("Low churn → build reference customer program")
        opps.append("Fix escalations → improve NPS")

        actions = [
            {"action":"Monthly health check-ins",
             "detail":f"Assign CSMs to all {profile['account_count']} accounts — catch churn signals early."},
            {"action":f"Address '{top_reason}' churn",
             "detail":"Build a targeted retention playbook for the top churn driver."},
            {"action":"Feature activation campaign",
             "detail":f"{feats:.0f} avg features — in-app nudges and onboarding can close the gap."},
        ]
        if mrr > 3000:
            actions.append({"action":"Upsell to Enterprise",
                            "detail":"Flag high-usage accounts for Enterprise upgrade conversations."})
        if esc > 0.5:
            actions.append({"action":"Resolve top escalations",
                            "detail":"Route most common escalation causes to the product roadmap."})

        segments.append({
            "segment_id":seg_id, "name":name,
            "profile":(
                f"<b>{profile['account_count']} accounts</b> · avg MRR <b>${mrr:,.0f}</b> · "
                f"churn <b>{churn}%</b> · satisfaction <b>{sat:.2f}/5</b> · "
                f"<b>{feats:.0f}</b> features · <b>{duration:.0f}hrs</b> usage · "
                f"mainly <b>{top_industry}</b> on <b>{top_plan}</b> plan."
            ),
            "risks":risks[:5], "opportunities":opps[:4],
            "recommended_actions":actions[:4],
            "health_score":health, "stats":s,
            "churn_rate":churn, "top_reason":top_reason,
            "account_count":profile["account_count"],
        })

    segments.sort(key=lambda x: x["segment_id"])
    total_churn = clustered_df["churn_flag"].mean()*100
    avg_mrr     = clustered_df["avg_mrr"].mean()
    worst = max(segments, key=lambda x: x["churn_rate"])
    best  = min(segments, key=lambda x: x["churn_rate"])

    return {
        "segments":segments, "worst":worst, "best":best,
        "total_churn":total_churn, "avg_mrr":avg_mrr,
        "summary":(
            f"<b>{len(clustered_df)} accounts</b> · avg MRR <b>${avg_mrr:,.0f}</b> · "
            f"overall churn <b>{total_churn:.1f}%</b>. "
            f"<b>{best['name']}</b> is your healthiest group — protect and expand it. "
            f"<b>{worst['name']}</b> has the highest churn and needs immediate action. "
            f"Activating low-engagement accounts and resolving escalation drivers "
            f"could meaningfully lift NRR this quarter."
        ),
        "priority":(
            f"Deploy an early warning system for <b>{worst['name']}</b> — "
            f"flag accounts with declining usage for CSM outreach before churn solidifies."
        ),
    }


# ─── Session state ───────────────────────────────────────────
for k,v in {"pipeline_run":False,"clustered_df":None,
             "profiles":None,"insights":None,"chat_history":[]}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div style='padding:16px 0 20px'>
  <div style='font-size:1.15rem;font-weight:800;color:{TEXT_H}'>RavenStack</div>
  <div style='font-size:0.62rem;font-weight:600;color:{TEXT_DIM};
              letter-spacing:0.14em;text-transform:uppercase;margin-top:3px'>
    Customer Insights Agent
  </div>
</div>""", unsafe_allow_html=True)
    st.divider()

    n_clusters = st.slider("Number of Segments", 2, 6, 3)
    run_btn    = st.button("▶  Run Full Pipeline", use_container_width=True, type="primary")

    st.divider()
    st.markdown("<div class='sec-header'>Pipeline</div>", unsafe_allow_html=True)
    s1=st.empty(); s2=st.empty(); s3=st.empty(); s4=st.empty()
    for s,t in [(s1,"Load data"),(s2,"Clean & encode"),(s3,"Cluster"),(s4,"Insights")]:
        s.markdown(f"<div style='font-size:0.8rem;color:{TEXT_DIM};padding:3px 0'>○ {t}</div>",
                   unsafe_allow_html=True)

    if st.session_state.pipeline_run:
        st.divider()
        df = st.session_state.clustered_df
        for lbl, val, color in [
            ("Total Accounts", f"{len(df):,}",                        "#7eb3f5"),
            ("Avg MRR",        f"${df['avg_mrr'].mean():,.0f}",       "#85cfc4"),
            ("Churn Rate",     f"{df['churn_flag'].mean()*100:.1f}%", "#f59ab3"),
            ("Segments",       str(df['segment'].nunique()),           "#b39af5"),
        ]:
            st.markdown(
                f"<div style='background:#fff;border:1px solid #e2e8f0;"
                f"border-left:3px solid {color};border-radius:10px;"
                f"padding:11px 14px;margin-bottom:8px'>"
                f"<div style='font-size:0.62rem;font-weight:600;color:#94a3b8;"
                f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px'>{lbl}</div>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#0f172a;line-height:1'>{val}</div>"
                f"</div>", unsafe_allow_html=True)


# ─── Pipeline runner ─────────────────────────────────────────
if run_btn:
    with st.spinner("Running pipeline..."):
        s1.markdown(f"<div style='font-size:0.8rem;color:{A_AMBER};padding:3px 0'>⏳ Loading...</div>", unsafe_allow_html=True)
        master = build_master_df()
        s1.markdown(f"<div style='font-size:0.8rem;color:{A_GREEN};padding:3px 0'>✓ Data loaded</div>", unsafe_allow_html=True)

        s2.markdown(f"<div style='font-size:0.8rem;color:{A_AMBER};padding:3px 0'>⏳ Cleaning...</div>", unsafe_allow_html=True)
        clean_df, scaled_df, scaler = clean(master)
        s2.markdown(f"<div style='font-size:0.8rem;color:{A_GREEN};padding:3px 0'>✓ Cleaned</div>", unsafe_allow_html=True)

        s3.markdown(f"<div style='font-size:0.8rem;color:{A_AMBER};padding:3px 0'>⏳ Clustering...</div>", unsafe_allow_html=True)
        clustered_df, profiles = run_clustering(clean_df, scaled_df, n_clusters=n_clusters)
        s3.markdown(f"<div style='font-size:0.8rem;color:{A_GREEN};padding:3px 0'>✓ Clustered</div>", unsafe_allow_html=True)

        s4.markdown(f"<div style='font-size:0.8rem;color:{A_AMBER};padding:3px 0'>⏳ Insights...</div>", unsafe_allow_html=True)
        insights = derive_insights(profiles, clustered_df)
        s4.markdown(f"<div style='font-size:0.8rem;color:{A_GREEN};padding:3px 0'>✓ Ready</div>", unsafe_allow_html=True)

    st.session_state.update({"pipeline_run":True, "clustered_df":clustered_df,
                              "profiles":profiles, "insights":insights})
    st.rerun()


# ─── Dashboard ───────────────────────────────────────────────
if st.session_state.pipeline_run:
    ins  = st.session_state.insights
    cdf  = st.session_state.clustered_df
    segs = ins["segments"]

    # Title
    st.markdown("<div class='page-title'>Customer Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>RavenStack · Agentic Analytics</div>", unsafe_allow_html=True)

    # KPI strip
    cols = st.columns(5)
    for col, (lbl, val, color) in zip(cols, [
        ("Total Accounts", f"{len(cdf):,}",                             P1),
        ("Avg MRR",        f"${cdf['avg_mrr'].mean():,.0f}",            A_GREEN),
        ("Churn Rate",     f"{cdf['churn_flag'].mean()*100:.1f}%",      A_RED),
        ("Avg Features",   f"{cdf['unique_features_used'].mean():.1f}", CHART_PAL[4]),
        ("Segments",       str(cdf["segment"].nunique()),                CHART_PAL[1]),
    ]):
        with col:
            st.markdown(f"""
<div class='kpi-card'>
  <div class='kpi-value' style='color:{color}'>{val}</div>
  <div class='kpi-label'>{lbl}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary — proper two-column via st.columns ────────────
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown(f"""
<div class='summary-card'>
  <div class='summary-card-label'>📊 Customer Base</div>
  <div class='summary-card-text'>{ins["summary"]}</div>
</div>""", unsafe_allow_html=True)

    with right_col:
        def _row(s):
            hc = hcolor(s["health_score"])
            nm = s["name"].split(" ", 1)[-1]
            hs = s["health_score"]
            cr = s["churn_rate"]
            mr = s["stats"].get("avg_mrr", 0)
            return (
                "<div class='seg-row'>"
                "<span class='seg-row-name'>" + nm + "</span>"
                "<span class='seg-row-stats'>"
                "<span style='color:" + hc + ";font-weight:700'>" + str(hs) + "/10</span>"
                "<span>" + str(cr) + "% churn</span>"
                "<span>$" + f"{mr:,.0f}" + " MRR</span>"
                "</span></div>"
            )
        rows = "".join(_row(s) for s in segs)
        st.markdown(f"""
<div class='summary-card'>
  <div class='summary-card-label'>⚡ Segment Snapshot</div>
  {rows}
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='prio-box'>🎯 <b>Top Priority —</b> {ins['priority']}</div>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Segment cards ─────────────────────────────────────────
    st.markdown("<div class='sec-header'>Segments</div>", unsafe_allow_html=True)

    # Use max 3 columns per row, wrap to next row if more segments
    n = len(segs)
    cols_per_row = min(n, 3)
    rows = [segs[i:i+cols_per_row] for i in range(0, n, cols_per_row)]

    for row in rows:
        rcols = st.columns(cols_per_row)
        for i, seg in enumerate(row):
            c = hcolor(seg["health_score"])
            with rcols[i]:
                st.markdown(f"""
<div class='seg-card'>
  <div class='seg-top-bar' style='background:{c}'></div>
  <div class='seg-name'>{seg["name"]}</div>
  <div class='seg-tag'>{seg["account_count"]} accounts · ${seg["stats"].get("avg_mrr",0):,.0f} avg MRR</div>
  <div style='display:flex;align-items:flex-end;justify-content:space-between;margin-top:10px'>
    <div>
      <span class='seg-health-num' style='color:{c}'>{seg["health_score"]}</span>
      <span class='seg-health-den'>/10</span>
      <div class='seg-health-lbl'>Health Score</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:1.3rem;font-weight:700;color:{A_RED}'>{seg["churn_rate"]}%</div>
      <div class='seg-health-lbl'>Churn Rate</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
        # Fill empty columns in last row
        for j in range(len(row), cols_per_row):
            with rcols[j]:
                st.empty()
        st.markdown("<br style='margin:4px'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analytics charts ──────────────────────────────────────
    st.markdown("<div class='sec-header'>Analytics Overview</div>", unsafe_allow_html=True)

    SEG_SHORT = {
        "High-Value Champions":    "Champions",
        "Engaged but At-Risk":     "At-Risk",
        "Low-Engagement Drifters": "Drifters",
        "Growth Potential":        "Growth",
        "Early Stage Accounts":    "Early Stage",
        "At-Risk Tail":            "At-Risk Tail",
    }
    def sname(s):
        full  = s["name"].split(" ",1)
        label = full[1] if len(full)>1 else s["name"]
        return SEG_SHORT.get(label, label[:12])

    c1,c2,c3 = st.columns(3)
    with c1:
        st.plotly_chart(bar_chart([sname(s) for s in segs],
            [round(s["stats"].get("avg_mrr",0)) for s in segs],
            CHART_PAL[0], "Avg MRR by Segment", "USD ($)"),
            use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.plotly_chart(bar_chart([sname(s) for s in segs],
            [s["churn_rate"] for s in segs],
            CHART_PAL[4], "Churn Rate by Segment", "Churn (%)"),
            use_container_width=True, config={"displayModeBar":False})
    with c3:
        st.plotly_chart(bar_chart([sname(s) for s in segs],
            [round(s["stats"].get("unique_features_used",0),1) for s in segs],
            P_DARK, "Feature Adoption by Segment", "Avg Features Used"),
            use_container_width=True, config={"displayModeBar":False})

    c4,c5,c6 = st.columns(3)
    with c4:
        fig = bar_chart([sname(s) for s in segs],
            [round(s["stats"].get("avg_satisfaction",0),2) for s in segs],
            CHART_PAL[3], "Satisfaction Score", "Score (out of 5)")
        fig.update_layout(yaxis=dict(range=[0,5.5]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with c5:
        ind = cdf["industry"].value_counts()
        st.plotly_chart(donut_chart(ind.index.tolist(), ind.values.tolist(),
            "Industry Distribution"),
            use_container_width=True, config={"displayModeBar":False})
    with c6:
        pl = cdf["plan_tier"].value_counts()
        st.plotly_chart(bar_chart(pl.index.tolist(), pl.values.tolist(),
            CHART_PAL[2], "Plan Distribution", "No. of Accounts"),
            use_container_width=True, config={"displayModeBar":False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Deep dive tabs ────────────────────────────────────────
    st.markdown("<div class='sec-header'>Deep Dive</div>", unsafe_allow_html=True)
    tabs = st.tabs([f"{hemoji(s['health_score'])}  {s['name']}" for s in segs])

    for tab, seg in zip(tabs, segs):
        with tab:
            seg_df    = cdf[cdf["segment"]==seg["segment_id"]]
            c_h       = hcolor(seg["health_score"])
            churn_val = seg_df["churn_flag"].mean()*100
            cc        = A_RED if churn_val>22 else A_AMBER if churn_val>18 else A_GREEN

            # KPI strip
            kpi_cols = st.columns(6)
            for kc,(lbl,val,col) in zip(kpi_cols,[
                ("Accounts",     str(len(seg_df)),                                 P1),
                ("Churn Rate",   f"{churn_val:.1f}%",                             cc),
                ("Avg MRR",      f"${seg_df['avg_mrr'].mean():,.0f}",             A_GREEN),
                ("Features Used",f"{seg_df['unique_features_used'].mean():.1f}",  CHART_PAL[4]),
                ("Satisfaction", f"{seg_df['avg_satisfaction'].mean():.2f} / 5",  CHART_PAL[1]),
                ("Health Score", f"{seg['health_score']} / 10",                   c_h),
            ]):
                with kc:
                    st.markdown(f"""
<div style='background:{SURFACE};border:1px solid {BORDER};border-top:3px solid {col};
            border-radius:10px;padding:14px 12px;text-align:center;
            box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
  <div style='font-size:1.4rem;font-weight:800;color:{col};line-height:1.1'>{val}</div>
  <div style='font-size:0.62rem;font-weight:600;color:{TEXT_DIM};
              text-transform:uppercase;letter-spacing:0.08em;margin-top:5px'>{lbl}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            left, right = st.columns(2)

            with left:
                st.markdown("**Profile**")
                st.markdown(f"<div class='profile-box'>{seg['profile']}</div>",
                            unsafe_allow_html=True)
                st.markdown("")
                st.markdown("**Risks**")
                st.markdown("<div style='margin-top:6px'>" +
                    "".join(f"<span class='pill-r'>{r}</span>" for r in seg["risks"]) +
                    "</div>", unsafe_allow_html=True)
                st.markdown("")
                st.markdown("**Opportunities**")
                st.markdown("<div style='margin-top:6px'>" +
                    "".join(f"<span class='pill-g'>{o}</span>" for o in seg["opportunities"]) +
                    "</div>", unsafe_allow_html=True)

            with right:
                st.markdown("**Recommended Actions**")
                for a in seg["recommended_actions"]:
                    st.markdown(f"""
<div class='action-item'>
  <div class='action-title'>{a['action']}</div>
  <div class='action-body'>{a['detail']}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            tc1,tc2,tc3 = st.columns(3)

            with tc1:
                ind = seg_df["industry"].value_counts()
                st.plotly_chart(hbar_chart(ind.index.tolist(), ind.values.tolist(),
                    CHART_PAL[0], "Industries", "Accounts", 220),
                    use_container_width=True, config={"displayModeBar":False})
            with tc2:
                pl = seg_df["plan_tier"].value_counts()
                st.plotly_chart(bar_chart(pl.index.tolist(), pl.values.tolist(),
                    P2, "Plans", "Accounts", 220),
                    use_container_width=True, config={"displayModeBar":False})
            with tc3:
                if "top_churn_reason" in seg_df.columns:
                    cr = seg_df["top_churn_reason"].replace("unknown",None).dropna().value_counts()
                    if len(cr):
                        st.plotly_chart(hbar_chart(cr.index.tolist(), cr.values.tolist(),
                            CHART_PAL[4], "Churn Reasons", "Accounts", 220),
                            use_container_width=True, config={"displayModeBar":False})

            st.markdown("**Account List**")
            display = ["account_name","industry","plan_tier","avg_mrr",
                       "unique_features_used","total_usage_duration_hrs","churn_flag","top_churn_reason"]
            avail = [c for c in display if c in seg_df.columns]
            st.dataframe(seg_df[avail].rename(columns={
                "account_name":"Account","industry":"Industry","plan_tier":"Plan",
                "avg_mrr":"Avg MRR ($)","unique_features_used":"Features",
                "total_usage_duration_hrs":"Usage (hrs)","churn_flag":"Churned",
                "top_churn_reason":"Churn Reason"}),
                use_container_width=True, height=260)

    # ── Chat ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>Ask the Agent</div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if user_query := st.chat_input("e.g. Which segment is most at risk?  What should I prioritise?"):
        st.session_state.chat_history.append({"role":"user","content":user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            q       = user_query.lower()
            worst   = ins["worst"]; best = ins["best"]
            richest = max(segs, key=lambda x: x["stats"].get("avg_mrr",0))
            most_f  = max(segs, key=lambda x: x["stats"].get("unique_features_used",0))

            if any(w in q for w in ["risk","churn","danger","urgent","worst","leaving","at-risk"]):
                p = worst
                response = (
                    f"### {hemoji(p['health_score'])} {p['name']} — Highest Risk\n\n"
                    f"**{p['churn_rate']}% churn** across **{p['account_count']} accounts** · Health **{p['health_score']}/10**\n\n"
                    f"**Primary churn reason:** {p['top_reason']}\n\n**Top actions:**\n" +
                    "\n".join(f"- **{a['action']}:** {a['detail']}" for a in p["recommended_actions"][:3])
                )
            elif any(w in q for w in ["best","healthy","strong","champion","safest"]):
                p = best
                response = (
                    f"### {hemoji(p['health_score'])} {p['name']} — Healthiest\n\n"
                    f"Only **{p['churn_rate']}% churn** · Health **{p['health_score']}/10** · ${p['stats'].get('avg_mrr',0):,.0f} avg MRR\n\n"
                    f"**Opportunities:**\n" + "\n".join(f"- {o}" for o in p["opportunities"][:3])
                )
            elif any(w in q for w in ["priorit","focus","quarter","strategy","what should","do next"]):
                response = (
                    f"### 🎯 Strategic Priorities\n\n"
                    f"{ins['priority'].replace('<b>','**').replace('</b>','**')}\n\n**By segment:**\n" +
                    "\n".join(f"- **{s['name']}** (health {s['health_score']}/10): {s['recommended_actions'][0]['action']}" for s in segs)
                )
            elif any(w in q for w in ["overview","breakdown","all","summary","how many"]):
                response = (
                    f"### 📊 Segment Overview\n\n"
                    f"**{len(cdf)} accounts** · **{ins['total_churn']:.1f}%** churn · **${ins['avg_mrr']:,.0f}** avg MRR\n\n" +
                    "\n".join(f"| {hemoji(s['health_score'])} **{s['name']}** | {s['account_count']} accounts | ${s['stats'].get('avg_mrr',0):,.0f} MRR | {s['churn_rate']}% churn | Health {s['health_score']}/10 |" for s in segs)
                )
            elif any(w in q for w in ["mrr","revenue","money","arr"]):
                response = (
                    f"### 💰 Revenue by Segment\n\nOverall avg MRR: **${ins['avg_mrr']:,.0f}**\n\n" +
                    "\n".join(f"- **{s['name']}**: ${s['stats'].get('avg_mrr',0):,.0f} avg MRR" for s in segs) +
                    f"\n\n**Highest value:** {richest['name']}"
                )
            elif any(w in q for w in ["feature","usage","adopt","engage"]):
                response = (
                    f"### 🖥️ Feature Adoption\n\n" +
                    "\n".join(f"- **{s['name']}**: {s['stats'].get('unique_features_used',0):.1f} features · {s['stats'].get('total_usage_duration_hrs',0):.0f}hrs" for s in segs) +
                    f"\n\n**Most engaged:** {most_f['name']}"
                )
            elif any(w in q for w in ["satisf","happy","nps","rating"]):
                response = (
                    f"### 😊 Satisfaction\n\n" +
                    "\n".join(f"- **{s['name']}**: {s['stats'].get('avg_satisfaction',0):.2f}/5" for s in segs) +
                    f"\n\nOverall: **{cdf['avg_satisfaction'].mean():.2f}/5**"
                )
            elif any(w in q for w in ["action","recommend","suggest","improve"]):
                response = "### ✅ Recommended Actions\n\n"
                for s in segs:
                    response += f"**{s['name']}**\n"
                    for a in s["recommended_actions"][:2]:
                        response += f"- **{a['action']}:** {a['detail']}\n"
                    response += "\n"
            elif any(w in q for w in ["upsell","upgrade","expand","opportunit"]):
                response = (
                    f"### 📈 Upsell Opportunities\n\nBest candidate: **{richest['name']}** — "
                    f"${richest['stats'].get('avg_mrr',0):,.0f} avg MRR\n\n" +
                    "\n".join(f"- {o}" for o in richest["opportunities"][:3])
                )
            else:
                response = (
                    f"### 📊 Overview\n\n**{len(cdf)} accounts** · **{ins['total_churn']:.1f}%** churn · **${ins['avg_mrr']:,.0f}** avg MRR\n\n"
                    f"🎯 {ins['priority'].replace('<b>','**').replace('</b>','**')}\n\n"
                    f"**Try asking:**\n- Which segment is most at risk?\n"
                    f"- What should I prioritise this quarter?\n"
                    f"- Show me revenue by segment\n"
                    f"- Which segment has the best feature adoption?\n"
                    f"- What are the upsell opportunities?"
                )

            st.markdown(response)
            st.session_state.chat_history.append({"role":"assistant","content":response})

else:
    st.markdown(f"""
<div style='display:flex;flex-direction:column;align-items:center;
            justify-content:center;min-height:78vh;text-align:center'>
  <div style='font-size:2.6rem;font-weight:800;color:{TEXT_H};margin-bottom:6px'>RavenStack</div>
  <div style='font-size:0.65rem;font-weight:600;color:{TEXT_DIM};
              letter-spacing:0.18em;text-transform:uppercase;margin-bottom:24px'>
    Customer Insights Agent
  </div>
  <div style='height:1px;width:48px;background:{BORDER};margin-bottom:24px'></div>
  <div style='color:{TEXT_DIM};font-size:0.88rem;line-height:1.9;max-width:340px'>
    Press <span style='color:{P1};font-weight:600'>Run Full Pipeline</span><br>
    in the sidebar to begin.
  </div>
</div>
""", unsafe_allow_html=True)