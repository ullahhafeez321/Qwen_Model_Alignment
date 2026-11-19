"""
human_evaluation.py – Human evaluation rating interface.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_merged_comparison,
    load_saved_ratings,
    save_ratings,
    load_human_ratings_template,
)

CRITERIA = ["helpfulness", "relevance", "clarity", "instruction_following"]
MODELS = ["BASE", "DPO_beta_0.1", "DPO_beta_0.5"]


def render():
    st.markdown("## ⭐ Human Evaluation (1–5 Rubric)")
    st.caption("Evaluate and score responses on Helpfulness, Relevance, Clarity, and Instruction Following.")

    if "ratings" not in st.session_state:
        st.session_state.ratings = load_saved_ratings()

    try:
        comparison = load_merged_comparison()
    except Exception:
        st.error("Cannot load comparison data.")
        return

    base_col = next((c for c in comparison.columns if "base" in c.lower()), None)
    b01_col = next((c for c in comparison.columns if "0.1" in c or "0_1" in c), None)
    b05_col = next((c for c in comparison.columns if "0.5" in c or "0_5" in c), None)

    col_map = {
        "BASE": base_col,
        "DPO_beta_0.1": b01_col,
        "DPO_beta_0.5": b05_col,
    }

    # Select prompt
    prompt_options = comparison.apply(lambda r: f"Prompt {r['id']} ({r['type']}): {r['prompt'][:60]}...", axis=1).tolist()
    selected_idx = st.selectbox("Select Prompt to Rate:", range(len(prompt_options)), format_func=lambda i: prompt_options[i])

    row = comparison.iloc[selected_idx]
    st.info(f"**Target Prompt:** {row['prompt']}")

    # Form
    with st.form(f"rating_form_{selected_idx}", clear_on_submit=False):
        ratings_data = []

        for model_name in MODELS:
            with st.container(border=True):
                st.markdown(f"### {_model_display(model_name)}")
                resp_col = col_map.get(model_name)
                resp_text = str(row.get(resp_col, "—")) if resp_col else "—"

                with st.expander("📄 View full response", expanded=True):
                    st.write(resp_text)

                cols = st.columns(4)
                scores = {}
                for i, criterion in enumerate(CRITERIA):
                    with cols[i]:
                        existing = _get_existing_rating(st.session_state.ratings, row["id"], model_name, criterion)
                        scores[criterion] = st.selectbox(
                            criterion.replace("_", " ").title(),
                            [1, 2, 3, 4, 5],
                            index=max(0, existing - 1) if existing > 0 else 2,
                            key=f"{selected_idx}_{model_name}_{criterion}",
                        )

                ratings_data.append({
                    "id": row["id"],
                    "type": row["type"],
                    "model": model_name,
                    **scores,
                    "notes": "",
                })

        submitted = st.form_submit_button("💾 Save Ratings for this Prompt", type="primary")
        if submitted:
            for entry in ratings_data:
                _upsert_rating(st.session_state.ratings, entry)
            save_ratings(st.session_state.ratings)
            st.success("✅ Ratings saved successfully!")

    # ── Summary of submitted ratings ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Submitted Ratings & Average Scores")

    ratings_df = st.session_state.ratings
    if ratings_df.empty or not _has_ratings(ratings_df):
        st.info("No ratings submitted yet. Score the prompts above to see live aggregated metrics.")
        with st.expander("📋 View blank university ratings template"):
            st.dataframe(load_human_ratings_template(), use_container_width=True)
    else:
        # Calculate averages
        numeric_rated = ratings_df.copy()
        for c in CRITERIA:
            numeric_rated[c] = pd.to_numeric(numeric_rated[c], errors="coerce")

        avg = numeric_rated.groupby("model")[CRITERIA].mean()
        avg["Overall Average"] = avg[CRITERIA].mean(axis=1)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.dataframe(avg.style.format("{:.2f}"), use_container_width=True)
        with c2:
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = {"BASE": "#2563EB", "DPO_beta_0.1": "#16A34A", "DPO_beta_0.5": "#EA580C"}
            for model_name in avg.index:
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=[c.replace("_", " ").title() for c in CRITERIA] + ["Overall"],
                    y=[avg.loc[model_name, c] for c in CRITERIA] + [avg.loc[model_name, "Overall Average"]],
                    marker_color=colors.get(model_name, "#64748B"),
                ))
            fig.update_layout(
                barmode="group",
                yaxis=dict(range=[1, 5], title="Score (1-5)"),
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        csv = ratings_df.to_csv(index=False)
        st.download_button("📥 Download Ratings CSV", csv, "human_ratings.csv", "text/csv")


def _model_display(name: str) -> str:
    return {
        "BASE": "🔵 Base Model (Qwen2.5-0.5B)",
        "DPO_beta_0.1": "🟢 DPO β=0.1",
        "DPO_beta_0.5": "🟠 DPO β=0.5",
    }.get(name, name)


def _get_existing_rating(df, prompt_id, model, criterion) -> int:
    if df.empty:
        return 0
    match = df[(df["id"] == prompt_id) & (df["model"] == model)]
    if match.empty:
        return 0
    val = match.iloc[0].get(criterion, 0)
    try:
        return int(val) if pd.notna(val) else 0
    except Exception:
        return 0


def _upsert_rating(df: pd.DataFrame, entry: dict) -> None:
    mask = (df["id"] == entry["id"]) & (df["model"] == entry["model"])
    if mask.any():
        for key, val in entry.items():
            df.loc[mask, key] = val
    else:
        new_row = pd.DataFrame([entry])
        st.session_state.ratings = pd.concat([df, new_row], ignore_index=True)


def _has_ratings(df: pd.DataFrame) -> bool:
    for c in CRITERIA:
        if c in df.columns and (pd.to_numeric(df[c], errors="coerce") > 0).any():
            return True
    return False
