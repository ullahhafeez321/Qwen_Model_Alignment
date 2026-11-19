"""
experiment_setup.py – Page 2: Experiment Setup details.
"""

import json
import streamlit as st
from utils.data_loader import (
    load_adapter_config,
    load_training_summary,
    load_training_metrics,
)
from utils.theme import metric_card


def render():
    st.markdown("# ⚙️ Experiment Setup")
    st.markdown(
        "All settings below are **verified from the saved artefacts** "
        "(adapter_config.json, training_metrics JSON, trainer_state.json)."
    )

    # ── Base model & dataset ─────────────────────────────────────────────
    st.markdown("### Base Model & Dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            | Parameter | Value |
            |---|---|
            | **Base Model** | `Qwen/Qwen2.5-0.5B` |
            | **Model Type** | Causal LM (text generation) |
            | **Framework** | Hugging Face Transformers + PEFT |
            | **Alignment Method** | Direct Preference Optimization (DPO) |
            """
        )
    with c2:
        st.markdown(
            """
            | Parameter | Value |
            |---|---|
            | **Preference Dataset** | `trl-lib/ultrafeedback_binarized` |
            | **Training Library** | Hugging Face TRL |
            | **Adapter Method** | LoRA (Low-Rank Adaptation) |
            | **Number of Training Runs** | 2 (beta 0.1, beta 0.5) |
            """
        )

    # ── LoRA configuration ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### LoRA Adapter Configuration")
    st.caption("Extracted from `adapter_config.json` – identical for both beta runs.")

    try:
        config = load_adapter_config("0_1")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("LoRA Rank (r)", str(config.get("r", "—"))), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("LoRA Alpha", str(config.get("lora_alpha", "—"))), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Dropout", str(config.get("lora_dropout", "—"))), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Bias", str(config.get("bias", "—"))), unsafe_allow_html=True)

        st.markdown("**Target Modules:**")
        modules = config.get("target_modules", [])
        st.code(", ".join(modules) if modules else "Not recorded", language=None)

        st.markdown("**Additional Configuration:**")
        extra = {
            "PEFT Type": config.get("peft_type", "—"),
            "Task Type": config.get("task_type", "—"),
            "Use DoRA": config.get("use_dora", "—"),
            "Use RSLoRA": config.get("use_rslora", "—"),
            "Inference Mode": config.get("inference_mode", "—"),
        }
        st.table(extra)

        with st.expander("📄 Raw adapter_config.json"):
            st.json(config)
    except Exception as exc:
        st.error(f"Could not load adapter config: {exc}")

    # ── DPO Beta Settings ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### DPO Training Runs")

    try:
        summary = load_training_summary()
        st.dataframe(
            summary.style.format({
                "train_runtime": "{:.2f}",
                "train_loss": "{:.6f}",
                "train_samples_per_second": "{:.3f}",
                "train_steps_per_second": "{:.3f}",
                "epoch": "{:.1f}",
            }),
            use_container_width=True,
        )
    except Exception:
        st.warning("training_metrics_summary.csv not found.")

    c1, c2 = st.columns(2)
    for col, beta_label, beta_key in [
        (c1, "β = 0.1", "0_1"),
        (c2, "β = 0.5", "0_5"),
    ]:
        with col:
            st.markdown(f"#### Training Run: {beta_label}")
            try:
                metrics = load_training_metrics(beta_key)
                st.markdown(f"- **Training Loss:** {metrics['train_loss']:.6f}")
                st.markdown(f"- **Runtime:** {metrics['train_runtime']:.2f} seconds")
                st.markdown(f"- **Samples/sec:** {metrics['train_samples_per_second']:.3f}")
                st.markdown(f"- **Steps/sec:** {metrics['train_steps_per_second']:.3f}")
                st.markdown(f"- **Epoch:** {metrics['epoch']}")
                with st.expander(f"📄 Raw metrics JSON ({beta_label})"):
                    st.json(metrics)
            except Exception as exc:
                st.error(f"Could not load metrics for {beta_label}: {exc}")

    st.markdown("---")
    st.info(
        "**Note on training settings:** Some training hyperparameters (learning rate "
        "schedule, warmup steps, gradient accumulation, max sequence length) are "
        "recorded in the trainer_state.json log history but not in a single summary "
        "field. The values shown above are those directly available from the saved "
        "artefacts. Any settings not shown here were not recorded in the available files."
    )

