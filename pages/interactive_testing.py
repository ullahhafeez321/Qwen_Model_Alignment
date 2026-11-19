"""
interactive_testing.py – Page 4: Live interactive inference with Qwen2.5-0.5B + DPO.
"""

import streamlit as st
from utils.model_loader import inference_available, generate_response
from utils.data_loader import load_base_outputs


PRESET_PROMPTS = [
    "--- Enter Custom Prompt ---",
    "Prompt 1 (Explanation): Explain supervised learning and reinforcement learning. Give one simple example of each.",
    "Prompt 2 (Advice): I have exactly three days to prepare for a programming exam. Create a realistic three-day study plan with breaks.",
    "Prompt 3 (Summarisation): Summarise this passage in exactly three bullet points: A database index is a data structure that helps a database find rows more quickly. It can improve SELECT queries, especially when searching or sorting by indexed columns. However, indexes require extra storage and can slow down INSERT, UPDATE, and DELETE operations because the index must also be maintained.",
    "Prompt 4 (Professional writing): Write a polite, concise email to a university instructor requesting a two-day extension for an assignment because of an unexpected personal emergency. Do not invent specific details.",
    "Prompt 5 (Reasoning): A student has 12 hours available. They spend one-third of the total time programming, then spend one-quarter of the remaining time revising. How many hours remain? Show the calculation.",
    "Prompt 6 (Question answering): What is the purpose of a primary key in a relational database? Answer in two or three sentences.",
    "Prompt 7 (Explanation): Explain the difference between HTTP and HTTPS in simple language. Mention what TLS does.",
    "Prompt 8 (Advice): A beginner wants to learn Python. Suggest a practical one-week learning plan with one main topic for each day.",
    "Prompt 9 (Summarisation): Summarise the following passage in exactly two sentences: Cloud computing provides computing services such as servers, storage, databases, networking, and software over the internet. Instead of buying and maintaining all their own hardware, users can access resources from a cloud provider and often pay according to usage. Cloud services can scale up or down as needs change.",
    "Prompt 10 (Reasoning): A shop sells a notebook for 200 rupees and gives a 25% discount. What is the discount amount and the final price? Show the arithmetic.",
]


def render():
    st.markdown("# 🧪 Interactive Prompt Testing")
    st.markdown(
        "Enter a custom prompt or choose an evaluation prompt to generate live responses "
        "from the **Base model (Qwen2.5-0.5B)**, **DPO β=0.1**, and **DPO β=0.5** in real time."
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

    st.markdown("---")

    # ── Preset Selection or Custom Prompt ────────────────────────────────
    c_preset, c_models = st.columns([2, 1])

    with c_preset:
        preset = st.selectbox(
            "Quick-select prompt or choose custom:",
            PRESET_PROMPTS,
            index=0,
        )

    # Determine default prompt text
    default_text = ""
    if preset != "--- Enter Custom Prompt ---":
        # Extract the prompt part after the colon
        parts = preset.split(": ", 1)
        if len(parts) > 1:
            default_text = parts[1]

    with c_models:
        model_choices = st.multiselect(
            "Select model(s) to evaluate:",
            ["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
            default=["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
        )

    prompt = st.text_area(
        "Prompt input:",
        value=default_text,
        height=130,
        placeholder="Enter your prompt here (e.g. 'Explain the difference between a list and a tuple in Python.')...",
    )

    # ── Hyperparameter Controls ──────────────────────────────────────────
    with st.expander("⚙️ Generation Hyperparameters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            max_tokens = st.slider("Max New Tokens", 32, 512, 192, step=32)
        with c2:
            temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)
        with c3:
            top_p = st.slider("Top-p Nucleus", 0.1, 1.0, 0.9, step=0.05)
        with c4:
            do_sample = st.checkbox("Sampling (uncheck for greedy)", value=True)

    model_key_map = {
        "Base (Qwen2.5-0.5B)": "base",
        "DPO β=0.1": "beta_0.1",
        "DPO β=0.5": "beta_0.5",
    }

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        generate_clicked = st.button("🚀 Generate Responses", type="primary", disabled=not prompt.strip())

    if generate_clicked and prompt.strip():
        st.markdown("---")
        st.info(
            "⚡ **Live generated responses** — generated in real time using the loaded weights. "
            "Prompt tokens are excluded from output decoding."
        )

        cols = st.columns(len(model_choices)) if model_choices else []

        for col, choice in zip(cols, model_choices):
            with col:
                st.markdown(f"### {choice}")
                key = model_key_map[choice]
                try:
                    with st.spinner(f"Generating with {choice}..."):
                        response_text, elapsed, token_cnt = generate_response(
                            prompt=prompt.strip(),
                            model_key=key,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                        )

                    speed = (token_cnt / elapsed) if elapsed > 0 else 0
                    st.caption(f"⏱️ Generated in {elapsed:.2f}s ({token_cnt} tokens, {speed:.1f} tok/s)")

                    st.markdown(
                        f'<div class="response-panel">{_escape(response_text)}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(
                        f"Generation error for {choice}: {exc}\n\n"
                        "Check console or logs for details."
                    )

    # ── Technical Architecture Note ──────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Technical Implementation Details"):
        st.markdown(
            """
            - **Unified Memory Architecture:** Base model `Qwen/Qwen2.5-0.5B` is loaded once into host memory.
              Both LoRA adapters (`beta_0.1` and `beta_0.5`) are registered onto the identical base network.
            - **Zero Redundant VRAM:** Base output is generated by calling `model.disable_adapter()`, eliminating duplicate model loading.
            - **Fair Comparison:** All selected models receive identical decoding parameters and attention masks.
            - **Token Boundary Control:** Generation halts cleanly at `<|endoftext|>` (151643) or `<|im_end|>` (151645).
            - **Inference Mode:** Executes within `torch.inference_mode()` with gradient computation disabled.
            """
        )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
