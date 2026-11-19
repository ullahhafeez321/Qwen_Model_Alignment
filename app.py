"""
app.py – Unified DPO Alignment Demonstration App (University LLM Lab 10).
"""

import streamlit as st

st.set_page_config(
    page_title="Qwen2.5 DPO Alignment | Lab 10",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.theme import inject_custom_css, render_footer

inject_custom_css()

# ── Sidebar: Model & Experiment Info Only (No broken page links) ─────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size: 2.2rem;">🔬</span>
            <h3 style="margin: 0.2rem 0; font-weight: 700;">DPO Alignment</h3>
            <span style="font-size: 0.85rem; opacity: 0.8;">University LLM Lab 10</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**Experiment Setup**")
        st.write("• **Base:** `Qwen/Qwen2.5-0.5B`")
        st.write("• **Dataset:** `UltraFeedback`")
        st.write("• **Method:** DPO + LoRA (r=16)")
        st.write("• **Runs:** β=0.1 vs β=0.5")

    st.markdown("---")
    st.markdown(
        """
        **Artifacts & Repositories:**  
        • [Hugging Face Model Repo](https://huggingface.co/ullahhafeez321/qwen-dpo-lab10)  
        • [GitHub Source Code](https://github.com/ullahhafeez321/Qwen_Model_Alignment)
        """
    )

# ── Main Header ──────────────────────────────────────────────────────────
st.title("🔬 Qwen2.5-0.5B Alignment: Direct Preference Optimization")

# ── Top-Level Tabs (Direct 1-Click Navigation) ───────────────────────────
tab_playground, tab_prompts, tab_training, tab_eval, tab_checklist = st.tabs([
    "🧪 Live Playground",
    "📊 10 Lab Prompts",
    "📈 Training Analysis",
    "⭐ Human Evaluation",
    "📋 Checklist & Report",
])

with tab_playground:
    from views.interactive_testing import render as render_playground
    render_playground()

with tab_prompts:
    from views.prompt_comparison import render as render_prompts
    render_prompts()

with tab_training:
    from views.training_analysis import render as render_training
    render_training()

with tab_eval:
    from views.human_evaluation import render as render_eval
    render_eval()

with tab_checklist:
    from views.checklist_report import render as render_checklist
    render_checklist()

render_footer()
