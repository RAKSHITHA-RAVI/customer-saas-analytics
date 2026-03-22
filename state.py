"""
state.py
--------
Defines the shared state object that flows through
every node in the LangGraph agent pipeline.
"""

from typing import TypedDict, Any, Optional


class AgentState(TypedDict):
    # Input
    user_query: str

    # Pipeline outputs (filled step by step)
    master_df:     Any             # raw merged dataframe
    clean_df:      Any             # cleaned dataframe
    scaled_df:     Any             # scaled features for clustering
    scaler:        Any             # fitted StandardScaler
    clustered_df:  Any             # df with 'segment' column
    profiles:      dict            # per-segment stats dict
    insights:      dict            # Claude API structured insights

    # Agent reasoning
    plan:          list[str]       # steps the planner decided to run
    completed:     list[str]       # steps finished so far
    errors:        list[str]       # any errors encountered
    final_answer:  Optional[str]   # human-readable final response

    # Config
    n_clusters:    Optional[int]   # override cluster count (None = auto)