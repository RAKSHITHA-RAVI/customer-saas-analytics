"""
orchestrator.py
---------------
LangGraph-based agentic orchestrator for the Customer Insights Agent.

Graph flow:
    planner → load_data → clean_data → cluster → generate_insights → responder → END

The planner reads the user query and decides which steps to run.
Each node is independently recoverable — errors are captured in state.

Usage:
    python orchestrator.py "Who are my highest risk customers?"
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

import anthropic
from langgraph.graph import StateGraph, END

from state import AgentState
from data_loader import build_master_df
from cleaner import clean
from clustering import run_clustering
from insight_generator import generate_segment_insights

client = anthropic.Anthropic()
MODEL  = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────────────────────
# NODE 1: Planner
# Reads the user query and decides which steps to run
# ─────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> AgentState:
    print("\n🧠 [Planner] Analysing query and building execution plan...")

    prompt = f"""
You are the planner for a SaaS customer analytics agent.

The agent has these tools available:
1. load_data          - Load and merge all customer data CSVs
2. clean_data         - Clean, encode and scale features
3. cluster            - Segment customers using KMeans
4. generate_insights  - Use Claude AI to name segments and produce recommendations
5. respond            - Synthesise a final answer for the user

Given this user query:
"{state['user_query']}"

Decide which steps to run. For most analytical queries, all 5 steps are needed.
Only skip steps if the query is clearly conversational and needs no data analysis.

Respond ONLY with a JSON array of step names in order.
Example: ["load_data", "clean_data", "cluster", "generate_insights", "respond"]
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    plan = json.loads(raw.strip())

    print(f"   📋 Plan: {' → '.join(plan)}")
    return {**state, "plan": plan, "completed": [], "errors": []}


# ─────────────────────────────────────────────────────────────
# NODE 2: Load Data
# ─────────────────────────────────────────────────────────────
def load_data_node(state: AgentState) -> AgentState:
    print("\n📦 [Load Data] Loading and merging CSVs...")
    try:
        master_df = build_master_df()
        print(f"   ✅ {master_df.shape[0]} accounts × {master_df.shape[1]} features")
        return {
            **state,
            "master_df": master_df,
            "completed": state["completed"] + ["load_data"],
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {**state, "errors": state["errors"] + [f"load_data: {e}"]}


# ─────────────────────────────────────────────────────────────
# NODE 3: Clean Data
# ─────────────────────────────────────────────────────────────
def clean_data_node(state: AgentState) -> AgentState:
    print("\n🧹 [Clean Data] Encoding and scaling features...")
    try:
        clean_df, scaled_df, scaler = clean(state["master_df"])
        print(f"   ✅ {clean_df.shape[1]} columns, 0 nulls, {scaled_df.shape[1]} scaled features")
        return {
            **state,
            "clean_df":  clean_df,
            "scaled_df": scaled_df,
            "scaler":    scaler,
            "completed": state["completed"] + ["clean_data"],
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {**state, "errors": state["errors"] + [f"clean_data: {e}"]}


# ─────────────────────────────────────────────────────────────
# NODE 4: Cluster
# ─────────────────────────────────────────────────────────────
def cluster_node(state: AgentState) -> AgentState:
    print("\n🤖 [Cluster] Segmenting customers...")
    try:
        n = state.get("n_clusters", None)
        clustered_df, profiles = run_clustering(
            state["clean_df"],
            state["scaled_df"],
            n_clusters=n
        )
        n_segs = clustered_df["segment"].nunique()
        print(f"   ✅ {n_segs} segments assigned to {len(clustered_df)} accounts")
        return {
            **state,
            "clustered_df": clustered_df,
            "profiles":     profiles,
            "completed":    state["completed"] + ["cluster"],
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {**state, "errors": state["errors"] + [f"cluster: {e}"]}


# ─────────────────────────────────────────────────────────────
# NODE 5: Generate Insights
# ─────────────────────────────────────────────────────────────
def generate_insights_node(state: AgentState) -> AgentState:
    print("\n💡 [Insights] Calling Claude API for segment analysis...")
    try:
        insights = generate_segment_insights(
            state["profiles"],
            state["clustered_df"],
            user_query=state["user_query"]
        )
        print("   ✅ Insights generated")
        return {
            **state,
            "insights":  insights,
            "completed": state["completed"] + ["generate_insights"],
        }
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {**state, "errors": state["errors"] + [f"generate_insights: {e}"]}


# ─────────────────────────────────────────────────────────────
# NODE 6: Responder
# Formats a clean final answer for the user
# ─────────────────────────────────────────────────────────────
def responder_node(state: AgentState) -> AgentState:
    print("\n✍️  [Responder] Composing final answer...")

    errors   = state.get("errors", [])
    insights = state.get("insights", {})

    if errors:
        answer = "⚠️ The agent encountered errors:\n" + "\n".join(f"  • {e}" for e in errors)
        return {**state, "final_answer": answer, "completed": state["completed"] + ["respond"]}

    segments = insights.get("segments", [])
    summary  = insights.get("executive_summary", "")
    priority = insights.get("top_priority_action", "")

    HEALTH_EMOJI = lambda s: "🔴" if s < 4 else ("🟡" if s < 7 else "🟢")

    lines = []
    lines.append(f"## Executive Summary\n{summary}")
    lines.append(f"\n🎯 **Top Priority:** {priority}")
    lines.append("\n---\n## Customer Segments\n")

    for seg in segments:
        emoji = HEALTH_EMOJI(seg["health_score"])
        lines.append(f"### {emoji} Segment {seg['segment_id']}: {seg['name']}  [Health {seg['health_score']}/10]")
        lines.append(f"_{seg['tagline']}_\n")
        lines.append(f"{seg['profile']}\n")
        lines.append("**Risks:** " + " · ".join(seg["risks"]))
        lines.append("\n**Opportunities:** " + " · ".join(seg["opportunities"]))
        lines.append("\n**Actions:**")
        for a in seg["recommended_actions"]:
            lines.append(f"  - **{a['action']}:** {a['detail']}")
        lines.append("")

    answer = "\n".join(lines)
    print("   ✅ Final answer ready")
    return {
        **state,
        "final_answer": answer,
        "completed":    state["completed"] + ["respond"],
    }


# ─────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner",           planner_node)
    graph.add_node("load_data",         load_data_node)
    graph.add_node("clean_data",        clean_data_node)
    graph.add_node("cluster",           cluster_node)
    graph.add_node("generate_insights", generate_insights_node)
    graph.add_node("respond",           responder_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",           "load_data")
    graph.add_edge("load_data",         "clean_data")
    graph.add_edge("clean_data",        "cluster")
    graph.add_edge("cluster",           "generate_insights")
    graph.add_edge("generate_insights", "respond")
    graph.add_edge("respond",           END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────
def run_agent(user_query: str, n_clusters: int = None) -> AgentState:
    """Main entry point. Run the full agent pipeline for a given query."""
    print("\n" + "=" * 60)
    print(f"🚀 AGENT START")
    print(f"   Query: {user_query}")
    print("=" * 60)

    graph = build_graph()

    initial_state: AgentState = {
        "user_query":   user_query,
        "master_df":    None,
        "clean_df":     None,
        "scaled_df":    None,
        "scaler":       None,
        "clustered_df": None,
        "profiles":     {},
        "insights":     {},
        "plan":         [],
        "completed":    [],
        "errors":       [],
        "final_answer": None,
        "n_clusters":   n_clusters,
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"✅ AGENT COMPLETE")
    print(f"   Steps: {' → '.join(final_state['completed'])}")
    if final_state["errors"]:
        print(f"   ⚠️  Errors: {final_state['errors']}")
    print("=" * 60)

    return final_state


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Who are my customer segments and what should I prioritise this quarter?"
    )
    result = run_agent(query)
    print("\n" + "─" * 60)
    print("📄 FINAL ANSWER")
    print("─" * 60)
    print(result["final_answer"])