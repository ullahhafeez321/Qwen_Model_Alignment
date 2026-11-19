"""
human_evaluation.py – Page 6: Human evaluation rating interface.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_merged_comparison,
    load_saved_ratings,
    save_ratings,
    load_human_ratings_template,
)
from utils.theme import metric_card


CRITERIA = ["helpfulness", "relevance", "clarity", "instruction_following"]
MODELS = ["BASE", "DPO_beta_0.1", "DPO_beta_0.5"]


def render():
    st.markdown("# ⭐ Human Evaluation")
    st.markdown(
        "Rate each model's response on a **1–5 scale** for the four evaluation "
        "criteria. Ratings are saved to a CSV file for later analysis."
    )

    # ── Template status ──────────────────────────────────────────────────
    with st.expander("📋 Original template (human_ratings_template.csv)"):
        st.caption(
            "This file is a **blank template** provided with the experiment. "
            "It does not contain completed ratings."
        )
        try:
            template = load_human_ratings_template()
            st.dataframe(template, use_container_width=True)
        except Exception:
            st.info("Template file not found.")

    st.markdown("---")

    # ── Load existing ratings ────────────────────────────────────────────
    if "ratings" not in st.session_state:
        st.session_state.ratings = load_saved_ratings()

    # ── Rating input ─────────────────────────────────────────────────────
    st.markdown("### Submit Ratings")

    try:
        comparison = load_merged_comparison()
    except Exception:
        st.error("Cannot load comparison data for rating.")
        return

    # Identify columns
    base_col = next((c for c in comparison.columns if "base" in c.lower()), None)
    b01_col = next((c for c in comparison.columns if "0.1" in c), None)
    b05_col = next((c for c in comparison.columns if "0.5" in c), None)

    col_map = {
        "BASE": base_col,
        "DPO_beta_0.1": b01_col,
        "DPO_beta_0.5": b05_col,
    }

    # Select prompt
    prompt_options = comparison.apply(
        lambda r: f"Prompt {r['id']} – {r['type']}", axis=1
    ).tolist()
    selected_idx = st.selectbox(
        "Select a prompt to rate:",
        range(len(prompt_options)),
        format_func=lambda i: prompt_options[i],
    )

    row = comparison.iloc[selected_idx]
    st.markdown(f"> **{row['prompt']}**")

    st.markdown("---")

    # Create rating form
    with st.form(f"rating_form_{selected_idx}", clear_on_submit=False):
        ratings_data = []

        for model_name in MODELS:
            st.markdown(f"#### {_model_display(model_name)}")
            resp_col = col_map.get(model_name)
            resp_text = str(row.get(resp_col, "—")) if resp_col else "—"

            with st.expander("View response", expanded=False):
                st.text(resp_text)

            cols = st.columns(4)
            scores = {}
            for i, criterion in enumerate(CRITERIA):
                with cols[i]:
                    existing = _get_existing_rating(
                        st.session_state.ratings,
                        row["id"], model_name, criterion,
                    )
                    scores[criterion] = st.selectbox(
                        criterion.replace("_", " ").title(),
                        [0, 1, 2, 3, 4, 5],
                        index=existing,
                        key=f"{selected_idx}_{model_name}_{criterion}",
                        help="0 = not rated, 1 = poor, 5 = excellent",
                    )

            notes = st.text_input(
                "Notes (optional)",
                key=f"notes_{selected_idx}_{model_name}",
                value=_get_existing_notes(
                    st.session_state.ratings, row["id"], model_name
                ),
            )

            ratings_data.append({
                "id": row["id"],
                "type": row["type"],
                "model": model_name,
                **scores,
                "notes": notes,
            })

        submitted = st.form_submit_button("💾 Save Ratings", type="primary")

        if submitted:
            for entry in ratings_data:
                # Only save if at least one criterion is rated (>0)
                if any(entry[c] > 0 for c in CRITERIA):
                    _upsert_rating(st.session_state.ratings, entry)

            save_ratings(st.session_state.ratings)
            st.success("Ratings saved successfully!")

    # ── Summary of submitted ratings ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### Submitted Ratings Summary")

    ratings_df = st.session_state.ratings
    rated = ratings_df[
        ratings_df[CRITERIA].apply(
            lambda r: any(r > 0), axis=1
        )
    ] if not ratings_df.empty else ratings_df

    if rated.empty:
        st.info(
            "No ratings have been submitted yet. Use the form above to rate "
            "each model's responses. Averages will be calculated from actual "
            "submitted ratings only."
        )
    else:
        st.dataframe(rated, use_container_width=True)

        # Calculate averages
        st.markdown("#### Average Scores by Model")
        numeric_rated = rated.copy()
        for c in CRITERIA:
            numeric_rated[c] = pd.to_numeric(numeric_rated[c], errors="coerce")
            numeric_rated.loc[numeric_rated[c] == 0, c] = pd.NA

        avg = numeric_rated.groupby("model")[CRITERIA].mean()
        if not avg.empty:
            st.dataframe(
                avg.style.format("{:.2f}", na_rep="—").background_gradient(
                    cmap="Blues", vmin=1, vmax=5
                ),
                use_container_width=True,
            )

            # Overall average
            avg["overall"] = avg[CRITERIA].mean(axis=1)

            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                colors = {"BASE": "#3182CE", "DPO_beta_0.1": "#38A169", "DPO_beta_0.5": "#DD6B20"}
                for model_name in avg.index:
                    fig.add_trace(go.Bar(
                        name=_model_display(model_name),
                        x=[c.replace("_", " ").title() for c in CRITERIA] + ["Overall"],
                        y=[avg.loc[model_name, c] for c in CRITERIA] + [avg.loc[model_name, "overall"]],
                        marker_color=colors.get(model_name, "#888"),
                    ))
                fig.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 5.5], title="Average Score (1–5)"),
                    template="plotly_white",
                    height=400,
                    margin=dict(t=30),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

        st.markdown("#### Average Scores by Task Type")
        avg_task = numeric_rated.groupby(["type", "model"])[CRITERIA].mean()
        if not avg_task.empty:
            st.dataframe(
                avg_task.style.format("{:.2f}", na_rep="—"),
                use_container_width=True,
            )

    # ── Download ratings ─────────────────────────────────────────────────
    st.markdown("---")
    if not rated.empty:
        csv = rated.to_csv(index=False)
        st.download_button(
            "📥 Download Ratings CSV",
            csv,
            "human_ratings.csv",
            "text/csv",
        )


def _model_display(name: str) -> str:
    return {
        "BASE": "🔵 Base Model",
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
    except (ValueError, TypeError):
        return 0


def _get_existing_notes(df, prompt_id, model) -> str:
    if df.empty:
        return ""
    match = df[(df["id"] == prompt_id) & (df["model"] == model)]
    if match.empty:
        return ""
    val = match.iloc[0].get("notes", "")
    return str(val) if pd.notna(val) else ""


def _upsert_rating(df: pd.DataFrame, entry: dict) -> None:
    """Update existing rating or append new one."""
    mask = (df["id"] == entry["id"]) & (df["model"] == entry["model"])
    if mask.any():
        for key, val in entry.items():
            df.loc[mask, key] = val
    else:
        new_row = pd.DataFrame([entry])
        st.session_state.ratings = pd.concat(
            [df, new_row], ignore_index=True
        )

