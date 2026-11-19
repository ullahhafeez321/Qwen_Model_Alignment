"""
prompt_comparison.py – Page 3: Side-by-side prompt comparison.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_merged_comparison
from utils.theme import badge


def render():
    st.markdown("# 📊 Prompt Comparison")
    st.markdown(
        "Compare the actual recorded responses from the **base model**, "
        "**DPO β=0.1**, and **DPO β=0.5** for each of the 10 evaluation prompts."
    )

    try:
        df = load_merged_comparison()
    except Exception as exc:
        st.error(f"Could not load comparison data: {exc}")
        return

    # Identify response columns
    base_col = [c for c in df.columns if "base" in c.lower()][0] if any("base" in c.lower() for c in df.columns) else None
    b01_col = [c for c in df.columns if "0.1" in c or "0_1" in c][0] if any("0.1" in c or "0_1" in c for c in df.columns) else None
    b05_col = [c for c in df.columns if "0.5" in c or "0_5" in c][0] if any("0.5" in c or "0_5" in c for c in df.columns) else None

    # ── Filters ──────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        task_types = ["All"] + sorted(df["type"].unique().tolist())
        selected_type = st.selectbox("Filter by task type", task_types)
    with col_f2:
        prompt_ids = ["All"] + sorted(df["id"].unique().tolist())
        selected_id = st.selectbox("Filter by prompt ID", prompt_ids)

    filtered = df.copy()
    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]
    if selected_id != "All":
        filtered = filtered[filtered["id"] == selected_id]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} prompts**")
    st.markdown("---")

    # ── Display each prompt ──────────────────────────────────────────────
    for _, row in filtered.iterrows():
        prompt_id = row["id"]
        task_type = row["type"]
        prompt_text = row["prompt"]

        st.markdown(
            f'### Prompt {prompt_id} '
            f'{badge(task_type, "blue")} '
            f'{badge(f"ID: {prompt_id}", "gray")}',
            unsafe_allow_html=True,
        )
        st.markdown(f"> **{prompt_text}**")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="response-panel">'
                f'<div class="response-panel-header">🔵 Base Model</div>'
                f'{_safe_text(row.get(base_col) if base_col else None)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📄 Full base response", expanded=False):
                st.text(str(row.get(base_col, "—")))

        with c2:
            st.markdown(
                f'<div class="response-panel">'
                f'<div class="response-panel-header">🟢 DPO β=0.1</div>'
                f'{_safe_text(row.get(b01_col) if b01_col else None)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📄 Full β=0.1 response", expanded=False):
                st.text(str(row.get(b01_col, "—")))

        with c3:
            st.markdown(
                f'<div class="response-panel">'
                f'<div class="response-panel-header">🟠 DPO β=0.5</div>'
                f'{_safe_text(row.get(b05_col) if b05_col else None)}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📄 Full β=0.5 response", expanded=False):
                st.text(str(row.get(b05_col, "—")))

        st.markdown("---")

    # ── Summary observations ─────────────────────────────────────────────
    st.markdown("### Observations from Recorded Outputs")
    st.warning(
        "**Important:** The three model configurations often produce **very similar or "
        "identical** responses for many prompts. This is consistent with the small model "
        "size (0.5B parameters), short training (1 epoch), and the nature of the LoRA "
        "adapter applying relatively small perturbations.\n\n"
        "Some outputs contain factual errors, hallucinations (e.g. 'Héldert Secure "
        "Hypertext Transfer ProtocoL'), arithmetic mistakes, and irrelevant "
        "continuations (e.g. appended classification prompts). These issues appear "
        "across all three configurations, indicating they originate from the base model "
        "rather than from DPO training."
    )


def _safe_text(val) -> str:
    """Return HTML-safe text, truncated for the panel view."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "<em>No response recorded</em>"
    text = str(val)
    # Escape basic HTML
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Truncate for panel display (full text is in expander)
    if len(text) > 600:
        text = text[:600] + "…"
    return text

