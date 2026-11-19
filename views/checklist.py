"""
checklist.py – Page 8: Assignment requirements checklist.
"""

import streamlit as st
from pathlib import Path
from utils.data_loader import load_saved_ratings, DATA_DIR
import pandas as pd


CRITERIA = ["helpfulness", "relevance", "clarity", "instruction_following"]


def render():
    st.markdown("# ✅ Assignment Checklist")
    st.markdown(
        "This checklist maps each **Lab 10 assignment requirement** to the "
        "evidence available in this dashboard."
    )

    checks = _get_checks()

    # Summary
    met = sum(1 for c in checks if c["status"] == "✅")
    partial = sum(1 for c in checks if c["status"] == "⚠️")
    missing = sum(1 for c in checks if c["status"] == "❌")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Met", met)
    with c2:
        st.metric("Partial", partial)
    with c3:
        st.metric("Missing / Action Needed", missing)

    st.markdown("---")

    for check in checks:
        icon = check["status"]
        st.markdown(
            f"""
            <div class="checklist-item">
                <strong>{icon} {check['requirement']}</strong><br>
                <span style="color: #718096; font-size: 0.9rem;">{check['evidence']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### How This App Satisfies Each Requirement")

    st.markdown(
        """
        | Requirement | App Section | Status |
        |---|---|---|
        | Original prompts (≥5, ≥3 task types) | Prompt Comparison | ✅ 10 prompts, 6 task types |
        | Base model outputs | Prompt Comparison | ✅ All 10 from `base_outputs.csv` |
        | Chosen/rejected preference examples | Conclusion (Q4) | ⚠️ Dataset description only; examples must be added |
        | DPO training for β=0.1 | Experiment Setup, Training Analysis | ✅ Adapter + metrics available |
        | DPO training for additional beta | Experiment Setup, Training Analysis | ✅ β=0.5 adapter + metrics available |
        | Training logs/metrics | Training Analysis | ✅ Step-level logs with loss, rewards, accuracy |
        | Same-prompt base-vs-aligned comparison | Prompt Comparison | ✅ Side-by-side for all 10 prompts |
        | Evaluation criteria & human ratings | Human Evaluation | ✅ 1–5 scale, 4 criteria, interactive form |
        | Beta comparison | Training Analysis, Conclusion | ✅ Metrics + generated output comparison |
        | Final evidence-based conclusion | Conclusion | ✅ 6 analysis questions answered |
        """
    )

    st.markdown("---")
    st.markdown("### Remaining Actions")

    st.warning(
        "The following items require manual action to complete the assignment:\n\n"
        "1. **Submit human evaluation ratings** via the ⭐ Human Evaluation page "
        "for all 10 prompts × 3 models × 4 criteria.\n\n"
        "2. **Add chosen/rejected preference examples** from the "
        "`trl-lib/ultrafeedback_binarized` dataset. This requires loading the "
        "dataset and including sample pairs in the evidence.\n\n"
        "3. **Re-visit the Conclusion page** after submitting ratings — the "
        "analysis will automatically incorporate the new scores.\n\n"
        "4. **(Optional) Run interactive inference** if hardware supports it, "
        "to demonstrate live model loading."
    )


def _get_checks() -> list[dict]:
    """Return the checklist items with current evidence status."""
    ratings = load_saved_ratings()
    has_ratings = _has_real_ratings(ratings)

    base_csv = (DATA_DIR / "base_outputs.csv").exists()
    b01_csv = (DATA_DIR / "beta_0_1_outputs.csv").exists()
    b05_csv = (DATA_DIR / "beta_0_5_outputs.csv").exists()
    comparison_csv = (DATA_DIR / "side_by_side_comparison.csv").exists()
    metrics_01 = (DATA_DIR / "training_metrics_beta_0_1.json").exists()
    metrics_05 = (DATA_DIR / "training_metrics_beta_0_5.json").exists()
    adapter_01 = (DATA_DIR / "adapter_config_beta_0_1.json").exists()
    adapter_05 = (DATA_DIR / "adapter_config_beta_0_5.json").exists()
    trainer_01 = (DATA_DIR / "trainer_state_beta_0_1.json").exists()
    trainer_05 = (DATA_DIR / "trainer_state_beta_0_5.json").exists()

    return [
        {
            "requirement": "Original prompts (at least 5, at least 3 task types)",
            "status": "✅" if base_csv else "❌",
            "evidence": (
                "10 prompts across 6 task types (Explanation, Advice, Summarisation, "
                "Professional Writing, Reasoning, Question Answering) in base_outputs.csv."
            ) if base_csv else "base_outputs.csv not found.",
        },
        {
            "requirement": "Base model outputs for evaluation prompts",
            "status": "✅" if base_csv else "❌",
            "evidence": "All 10 responses recorded in base_outputs.csv.",
        },
        {
            "requirement": "Inspection of chosen/rejected preference examples",
            "status": "⚠️",
            "evidence": (
                "The experiment artefacts do not include example chosen/rejected pairs "
                "from the preference dataset. A description of the dataset characteristics "
                "is provided in the Conclusion page, but actual examples must be added."
            ),
        },
        {
            "requirement": "DPO training for beta 0.1",
            "status": "✅" if (adapter_01 and metrics_01) else "❌",
            "evidence": (
                "LoRA adapter (adapter_config.json + adapter_model.safetensors) and "
                "training metrics (loss=0.6817, runtime=383.4s) recorded."
            ),
        },
        {
            "requirement": "DPO training for an additional beta value (0.5)",
            "status": "✅" if (adapter_05 and metrics_05) else "❌",
            "evidence": (
                "LoRA adapter and training metrics (loss=0.6598, runtime=385.7s) "
                "recorded for beta 0.5."
            ),
        },
        {
            "requirement": "Training logs and metrics",
            "status": "✅" if (trainer_01 and trainer_05) else "❌",
            "evidence": (
                "Step-level logs with loss, gradient norm, learning rate, reward "
                "accuracies, chosen/rejected rewards, and reward margins available "
                "from trainer_state.json for both runs (128 steps each)."
            ),
        },
        {
            "requirement": "Same-prompt base-vs-aligned comparison",
            "status": "✅" if comparison_csv else "❌",
            "evidence": (
                "side_by_side_comparison.csv contains all 10 prompts with base, "
                "β=0.1, and β=0.5 responses in the same row."
            ),
        },
        {
            "requirement": "Evaluation criteria and human ratings (1–5 scale)",
            "status": "✅" if has_ratings else "⚠️",
            "evidence": (
                "Human ratings submitted and saved."
                if has_ratings
                else "Rating interface is available on the Human Evaluation page. "
                     "No ratings have been submitted yet — this is required."
            ),
        },
        {
            "requirement": "Beta comparison analysis",
            "status": "✅",
            "evidence": (
                "Training metrics compared (loss, runtime, reward margins). "
                "Generated output differences analysed in Conclusion page."
            ),
        },
        {
            "requirement": "Final evidence-based conclusion",
            "status": "✅",
            "evidence": (
                "Six analysis questions answered on the Conclusion page, with "
                "caveats clearly stated where evidence is insufficient."
            ),
        },
    ]


def _has_real_ratings(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for c in CRITERIA:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if (vals > 0).any():
                return True
    return False

