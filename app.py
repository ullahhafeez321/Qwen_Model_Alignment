"""
app.py – Main entry point for the DPO Lab 10 Alignment Dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="DPO Alignment Lab 10",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import inject_custom_css, render_footer  # noqa: E402

inject_custom_css()

# ── Sidebar navigation ──────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size:2rem;">🔬</span><br>
        <strong style="font-size:1.1rem; color:#1E3A5F;">DPO Lab 10</strong><br>
        <span style="font-size:0.8rem; color:#718096;">Alignment Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "⚙️ Experiment Setup",
        "📊 Prompt Comparison",
        "🧪 Interactive Testing",
        "📈 Training Analysis",
        "⭐ Human Evaluation",
        "📝 Conclusion",
        "✅ Assignment Checklist",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Base model:** Qwen/Qwen2.5-0.5B  \n"
    "**Method:** DPO with LoRA  \n"
    "**Beta values:** 0.1 & 0.5"
)

# ── Page routing ─────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from pages.overview import render
    render()
elif page == "⚙️ Experiment Setup":
    from pages.experiment_setup import render
    render()
elif page == "📊 Prompt Comparison":
    from pages.prompt_comparison import render
    render()
elif page == "🧪 Interactive Testing":
    from pages.interactive_testing import render
    render()
elif page == "📈 Training Analysis":
    from pages.training_analysis import render
    render()
elif page == "⭐ Human Evaluation":
    from pages.human_evaluation import render
    render()
elif page == "📝 Conclusion":
    from pages.conclusion import render
    render()
elif page == "✅ Assignment Checklist":
    from pages.checklist import render
    render()

render_footer()

