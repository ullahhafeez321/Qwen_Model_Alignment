"""
prompt_comparison.py – Side-by-side prompt comparison with search & filter.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_merged_comparison
from utils.theme import badge


def render():
    st.markdown("## 📊 10 Lab Prompts (Side-by-Side Comparison)")
    st.caption("Compare recorded outputs from the Base model, DPO β=0.1, and DPO β=0.5 across all 10 university prompts.")

    try:
        df = load_merged_comparison()
    except Exception as exc:
        st.error(f"Could not load comparison data: {exc}")
        return

    base_col = next((c for c in df.columns if "base" in c.lower()), None)
    b01_col = next((c for c in df.columns if "0.1" in c or "0_1" in c), None)
    b05_col = next((c for c in df.columns if "0.5" in c or "0_5" in c), None)

    # ── Search & Filters ─────────────────────────────────────────────────
    col_search, col_type, col_id = st.columns([2, 1, 1])
    with col_search:
        search_query = st.text_input("🔍 Search prompts or responses:", placeholder="Type keywords (e.g. database, HTTP, email)...")
    with col_type:
        task_types = ["All Types"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Filter by Task:", task_types)
    with col_id:
        prompt_ids = ["All Prompts"] + [f"Prompt {pid}" for pid in sorted(df["id"].unique().tolist())]
        selected_id_str = st.selectbox("Prompt ID:", prompt_ids)

    # Filter data
    filtered = df.copy()
    if selected_type != "All Types":
        filtered = filtered[filtered["type"] == selected_type]
    if selected_id_str != "All Prompts":
        target_id = int(selected_id_str.replace("Prompt ", ""))
        filtered = filtered[filtered["id"] == target_id]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[
            filtered["prompt"].str.lower().str.contains(q, na=False) |
            filtered[base_col].astype(str).str.lower().str.contains(q, na=False) |
            filtered[b01_col].astype(str).str.lower().str.contains(q, na=False) |
            filtered[b05_col].astype(str).str.lower().str.contains(q, na=False)
        ]

    st.write(f"Showing **{len(filtered)}** of {len(df)} prompts")

    # ── Display Prompts ──────────────────────────────────────────────────
    for _, row in filtered.iterrows():
        prompt_id = row["id"]
        task_type = row["type"]
        prompt_text = row["prompt"]

        with st.container(border=True):
            st.markdown(
                f"**Prompt {prompt_id}** {badge(task_type, 'blue')}",
                unsafe_allow_html=True,
            )
            st.info(f"**Question:** {prompt_text}")

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.markdown("#### 🔵 Base Model")
                    resp = str(row.get(base_col, "—"))
                    st.write(resp)

            with c2:
                with st.container(border=True):
                    st.markdown("#### 🟢 DPO β=0.1")
                    resp = str(row.get(b01_col, "—"))
                    st.write(resp)

            with c3:
                with st.container(border=True):
                    st.markdown("#### 🟠 DPO β=0.5")
                    resp = str(row.get(b05_col, "—"))
                    st.write(resp)
