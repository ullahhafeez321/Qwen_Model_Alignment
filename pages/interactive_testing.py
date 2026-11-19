"""
interactive_testing.py – Page 4: Live inference with the loaded models.
"""

import streamlit as st
from utils.model_loader import inference_available


def render():
    st.markdown("# 🧪 Interactive Prompt Testing")
    st.markdown(
        "Enter a custom prompt and generate responses from the base model "
        "and/or DPO-aligned adapters **in real time**."
    )

    available, reason = inference_available()

    if not available:
        st.warning(
            f"**Live inference is unavailable** on this host.  \n"
            f"Reason: {reason}  \n\n"
            "The rest of the dashboard (saved results, training analysis, human "
            "evaluation) remains fully functional without model weights.  \n\n"
            "To enable live inference, install the required dependencies:\n"
            "```\npip install torch transformers peft huggingface_hub\n```"
        )
        st.info(
            "💡 **Tip:** To view saved experiment responses, navigate to the "
            "**Prompt Comparison** page."
        )
        return

    # ── Inference UI ─────────────────────────────────────────────────────
    st.markdown("---")

    prompt = st.text_area(
        "Enter your prompt:",
        height=120,
        placeholder="e.g. Explain the difference between a list and a tuple in Python.",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        model_choices = st.multiselect(
            "Select model(s):",
            ["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
            default=["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
        )
    with col2:
        max_tokens = st.slider("Max new tokens", 64, 512, 256, step=64)

    model_key_map = {
        "Base (Qwen2.5-0.5B)": "base",
        "DPO β=0.1": "beta_0.1",
        "DPO β=0.5": "beta_0.5",
    }

    generate = st.button("🚀 Generate", type="primary", disabled=not prompt.strip())

    if generate and prompt.strip():
        st.markdown("---")
        st.info(
            "⚡ **Live-generated responses** — these are produced in real time "
            "and may differ from the saved experiment outputs."
        )

        cols = st.columns(len(model_choices)) if model_choices else []

        for col, choice in zip(cols, model_choices):
            with col:
                st.markdown(f"**{choice}**")
                key = model_key_map[choice]
                try:
                    with st.spinner(f"Generating with {choice}…"):
                        from utils.model_loader import generate_response
                        response = generate_response(
                            prompt.strip(),
                            model_key=key,
                            max_new_tokens=max_tokens,
                        )
                    st.markdown(
                        f'<div class="response-panel">{_escape(response)}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(
                        f"Generation failed for {choice}: {exc}\n\n"
                        "This may be due to insufficient memory or missing "
                        "adapter weights. The dashboard remains functional."
                    )

    # ── Explanation panel ────────────────────────────────────────────────
    with st.expander("ℹ️ How live inference works"):
        st.markdown(
            """
            - The **base model** (`Qwen/Qwen2.5-0.5B`) is loaded once and cached.
            - Each **DPO adapter** is loaded from the HuggingFace repository
              (`ullahhafeez321/qwen-dpo-lab10`) as a PEFT LoRA adapter on top
              of the base model.
            - **Identical generation settings** (temperature, top-p, max tokens)
              are used for all three variants.
            - Only **newly generated tokens** are shown (the prompt is excluded
              from the output).
            - The model runs in **eval mode** with `torch.inference_mode()`.
            - Device is auto-detected (CUDA → MPS → CPU).
            """
        )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

