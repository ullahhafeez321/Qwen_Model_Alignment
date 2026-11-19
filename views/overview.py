"""
overview.py – Page 1: Overview of the DPO Alignment Experiment.
"""

import streamlit as st
from utils.theme import metric_card, badge


def render():
    st.markdown("# 🏠 Overview")
    st.markdown(
        "### Direct Preference Optimization (DPO) Alignment Experiment"
    )

    st.markdown(
        """
        This dashboard presents the results of a university **Large Language Models Lab 10**
        experiment investigating whether **Direct Preference Optimization (DPO)** can
        meaningfully change the behaviour of a pretrained language model.

        DPO is a preference-based alignment method that trains a model to favour responses
        that humans prefer (the *chosen* responses) over those they dislike (the *rejected*
        responses), using a dataset of paired examples. Unlike RLHF, DPO does not need a
        separate reward model—it optimises the policy directly through a classification-style
        loss derived from the Bradley–Terry preference model.
        """
    )

    # ── Key experiment parameters ────────────────────────────────────────
    st.markdown("### Experiment at a Glance")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Base Model", "Qwen2.5-0.5B"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Alignment Method", "DPO + LoRA"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Beta Values", "0.1  &  0.5"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Eval Prompts", "10"), unsafe_allow_html=True)

    st.markdown("")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(metric_card("Preference Dataset", "UltraFeedback"), unsafe_allow_html=True)
    with c6:
        st.markdown(metric_card("Adapter", "LoRA (r=16)"), unsafe_allow_html=True)
    with c7:
        st.markdown(metric_card("Training Epochs", "1"), unsafe_allow_html=True)
    with c8:
        st.markdown(metric_card("Task Types", "6"), unsafe_allow_html=True)

    # ── Beta explanation ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Understanding the DPO Beta Parameter")
    st.info(
        "**Beta (β)** controls the strength of the KL-divergence constraint between "
        "the trained policy and the reference (base) model in the DPO loss function.\n\n"
        "- **Lower beta (e.g. 0.1):** The optimiser is allowed to diverge more from "
        "the reference model. The model can change its behaviour more aggressively to "
        "match preferences, but risks overfitting to the preference dataset or "
        "deviating too far from sensible base-model behaviour.\n\n"
        "- **Higher beta (e.g. 0.5):** The KL penalty is stronger, keeping the trained "
        "model closer to the reference. This can preserve more of the base model's "
        "general capabilities at the expense of less aggressive preference learning.\n\n"
        "Beta does **not** directly control output quality; it controls the trade-off "
        "between preference alignment strength and conservatism relative to the "
        "base model."
    )

    # ── Experiment task types ────────────────────────────────────────────
    st.markdown("### Evaluation Task Types")
    st.markdown(
        "The 10 evaluation prompts span **6 task categories** to test different "
        "aspects of model capability:"
    )
    tasks = {
        "Explanation": "Explain technical concepts clearly (2 prompts)",
        "Advice": "Provide actionable, structured guidance (2 prompts)",
        "Summarisation": "Condense information accurately (2 prompts)",
        "Professional Writing": "Draft polished written content (1 prompt)",
        "Reasoning": "Solve problems step by step (2 prompts)",
        "Question Answering": "Answer factual questions concisely (1 prompt)",
    }
    cols = st.columns(3)
    for i, (task, desc) in enumerate(tasks.items()):
        with cols[i % 3]:
            st.markdown(f"**{task}**")
            st.caption(desc)

    # ── Comparison overview ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Three-Way Comparison")
    st.markdown(
        """
        The dashboard compares responses from three model configurations using
        **exactly the same prompts**:

        | Configuration | Description |
        |---|---|
        | **Base** | Qwen2.5-0.5B with no alignment (pretrained weights only) |
        | **DPO β=0.1** | Base + LoRA adapter trained with DPO (β=0.1) |
        | **DPO β=0.5** | Base + LoRA adapter trained with DPO (β=0.5) |

        Navigate to **Prompt Comparison** to inspect the actual responses, or
        **Training Analysis** to see the recorded training metrics.
        """
    )

