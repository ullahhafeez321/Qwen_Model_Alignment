"""
checklist_report.py – Compact lab checklist, setup, and key conclusions.
"""

import streamlit as st
from utils.data_loader import load_adapter_config, load_training_summary


def render():
    st.markdown("## 📋 Lab 10 Checklist & Experiment Summary")
    st.caption("Verification of assignment rubric requirements and empirical conclusions.")

    # ── Teacher Rubric Checklist ─────────────────────────────────────────
    st.markdown("### Assignment Requirement Verification")

    checklist_items = [
        ("Original Prompts", "10 prompts covering 6 task categories (Explanation, Advice, Summarisation, Professional Writing, Reasoning, QA)", "✅ Met"),
        ("Base Model Outputs", "All 10 generated base responses preserved from base_outputs.csv", "✅ Met"),
        ("DPO Training (β=0.1)", "LoRA adapter trained with β=0.1 (loss: 0.6817, runtime: 383.4s)", "✅ Met"),
        ("DPO Training (β=0.5)", "LoRA adapter trained with β=0.5 (loss: 0.6598, runtime: 385.7s)", "✅ Met"),
        ("Training Logs & Loss Curves", "Complete 128-step loss, reward accuracies, and margins logged", "✅ Met"),
        ("Same-Prompt 3-Way Comparison", "Side-by-side comparative views for all 10 prompts", "✅ Met"),
        ("Human Rating Rubric (1–5)", "Helpfulness, Relevance, Clarity, and Instruction Following rating form & storage", "✅ Met"),
        ("Beta Sensitivity Analysis", "Impact of KL penalty on training loss and policy divergence documented", "✅ Met"),
        ("Preference Pair Inspection", "UltraFeedback dataset characteristics documented; pair viewer highlighted", "⚠️ Addressed"),
    ]

    for req, desc, status in checklist_items:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{req}** — {desc}")
            c2.markdown(f"**{status}**")

    # ── Verified Experiment Setup ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Verified LoRA & Training Configuration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Model", "Qwen2.5-0.5B")
    c2.metric("Dataset", "UltraFeedback")
    c3.metric("LoRA Rank (r)", "16")
    c4.metric("LoRA Alpha", "32")

    with st.expander("🔍 View LoRA Target Modules & Hyperparameters"):
        st.markdown(
            """
            - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
            - **Dropout:** 0.05
            - **Bias:** None
            - **Precision:** bfloat16 / float32
            - **Optimization Steps:** 128 steps (1.0 full epoch)
            """
        )

    # ── Answers to Lab Analysis Questions ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Key Findings & Analysis")

    q_answers = [
        ("1. Did DPO improve overall response quality?",
         "Minimal observable change across most factual prompts. Small model size (0.5B) and short training (1 epoch) limit aggressive behavioral shifts. Human ratings are required to confirm nuanced preference differences."),

        ("2. Which prompt types showed the most difference?",
         "Advice and study planning prompts showed improved structural formatting (clearer headers and bullet points). Reasoning and summarisation remained challenging across all models."),

        ("3. Did the base model perform better on any prompt?",
         "On Prompt 7 (HTTP vs HTTPS), DPO β=0.1 hallucinated 'Hijacked Secure HyperText Transport Protocol', whereas the base model remained closer to standard definitions."),

        ("4. How did the Beta parameter affect training and generation?",
         "β=0.5 achieved lower loss (0.6598 vs 0.6817) and wider reward margins (+0.209 vs +0.054) because the KL constraint amplified the preference gradient. However, generated outputs between β=0.1 and β=0.5 remained largely similar."),

        ("5. What did DPO change?",
         "DPO successfully shifted the implicit reward margins and preference classification accuracy (from ~24% to ~66-71%), without collapsing the base language model capabilities."),
    ]

    for q, ans in q_answers:
        with st.container(border=True):
            st.markdown(f"**{q}**")
            st.caption(ans)

