"""
interactive_testing.py – Live & preset model prompt testing playground.
"""

import streamlit as st
from utils.model_loader import inference_available, generate_response
from utils.data_loader import load_merged_comparison

LAB_PROMPTS = {
    "Explain supervised vs reinforcement learning": "Explain supervised learning and reinforcement learning. Give one simple example of each.",
    "3-day study plan with breaks": "I have exactly three days to prepare for a programming exam. Create a realistic three-day study plan with breaks.",
    "Database index 3-bullet summary": "Summarise this passage in exactly three bullet points: A database index is a data structure that helps a database find rows more quickly. It can improve SELECT queries, especially when searching or sorting by indexed columns. However, indexes require extra storage and can slow down INSERT, UPDATE, and DELETE operations because the index must also be maintained.",
    "Polite email requesting extension": "Write a polite, concise email to a university instructor requesting a two-day extension for an assignment because of an unexpected personal emergency. Do not invent specific details.",
    "Time math reasoning problem": "A student has 12 hours available. They spend one-third of the total time programming, then spend one-quarter of the remaining time revising. How many hours remain? Show the calculation.",
    "Primary key definition": "What is the purpose of a primary key in a relational database? Answer in two or three sentences.",
    "HTTP vs HTTPS & TLS": "Explain the difference between HTTP and HTTPS in simple language. Mention what TLS does.",
    "1-week Python learning plan": "A beginner wants to learn Python. Suggest a practical one-week learning plan with one main topic for each day.",
    "Cloud computing 2-sentence summary": "Summarise the following passage in exactly two sentences: Cloud computing provides computing services such as servers, storage, databases, networking, and software over the internet. Instead of buying and maintaining all their own hardware, users can access resources from a cloud provider and often pay according to usage. Cloud services can scale up or down as needs change.",
    "Discount arithmetic problem": "A shop sells a notebook for 200 rupees and gives a 25% discount. What is the discount amount and the final price? Show the arithmetic.",
}


def render():
    st.markdown("## 🧪 Live Model Playground")
    st.markdown(
        '<div class="page-subtitle">'
        'Type any custom prompt or choose a lab prompt to compare outputs from '
        '<b>Base (Qwen2.5-0.5B)</b>, <b>DPO β=0.1</b>, and <b>DPO β=0.5</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

    live_ready, reason = inference_available()

    if not live_ready:
        st.info("ℹ️ **Cloud Environment Note:** PyTorch runtime is currently building on the host. You can test all 10 university prompts below with instant experiment outputs, or live generation will activate automatically once dependencies finish building.")

    # ── Top Controls ─────────────────────────────────────────────────────
    col_preset, col_models = st.columns([1.2, 1])

    with col_preset:
        preset_choice = st.selectbox(
            "Quick-select prompt:",
            ["[Custom Prompt]"] + list(LAB_PROMPTS.keys()),
            index=1 if not live_ready else 0,
        )

    initial_prompt = "" if preset_choice == "[Custom Prompt]" else LAB_PROMPTS[preset_choice]

    with col_models:
        selected_models = st.multiselect(
            "Models to compare:",
            ["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
            default=["Base (Qwen2.5-0.5B)", "DPO β=0.1", "DPO β=0.5"],
        )

    user_prompt = st.text_area(
        "Prompt:",
        value=initial_prompt,
        height=100,
        placeholder="Type any prompt here...",
    )

    col_btn, col_len, col_temp = st.columns([1.2, 1, 1])
    with col_btn:
        st.write("")
        generate = st.button("🚀 Generate Responses", type="primary", use_container_width=True, disabled=not user_prompt.strip())
    with col_len:
        max_tokens = st.slider("Max new tokens", 32, 384, 160, step=32)
    with col_temp:
        temperature = st.slider("Temperature", 0.1, 1.2, 0.7, step=0.1)

    # ── Handle Output ────────────────────────────────────────────────────
    if generate and user_prompt.strip():
        if not selected_models:
            st.warning("Please select at least one model.")
            return

        cols = st.columns(len(selected_models))
        model_key_map = {
            "Base (Qwen2.5-0.5B)": "base",
            "DPO β=0.1": "beta_0.1",
            "DPO β=0.5": "beta_0.5",
        }

        # Check if prompt matches one of the saved lab prompts
        df_saved = load_merged_comparison()
        base_col = next((c for c in df_saved.columns if "base" in c.lower()), None)
        b01_col = next((c for c in df_saved.columns if "0.1" in c or "0_1" in c), None)
        b05_col = next((c for c in df_saved.columns if "0.5" in c or "0_5" in c), None)

        matched_row = None
        for _, row in df_saved.iterrows():
            if row["prompt"].strip().lower() in user_prompt.strip().lower() or user_prompt.strip().lower() in row["prompt"].strip().lower():
                matched_row = row
                break

        for col, model_label in zip(cols, selected_models):
            with col:
                with st.container(border=True):
                    if "Base" in model_label:
                        st.markdown(f"### 🔵 {model_label}")
                    elif "0.1" in model_label:
                        st.markdown(f"### 🟢 {model_label}")
                    else:
                        st.markdown(f"### 🟠 {model_label}")

                    key = model_key_map[model_label]

                    if live_ready:
                        with st.spinner(f"Generating live..."):
                            try:
                                resp, elapsed, tok_cnt = generate_response(
                                    prompt=user_prompt.strip(),
                                    model_key=key,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=0.9,
                                    do_sample=True,
                                )
                                speed = (tok_cnt / elapsed) if elapsed > 0 else 0
                                st.caption(f"⚡ Live generated in {elapsed:.1f}s ({tok_cnt} tokens, {speed:.0f} tok/s)")
                                st.write(resp)
                            except Exception as e:
                                st.error(f"Live generation error: {e}")
                    else:
                        # Fallback to saved experiment response if prompt matches
                        if matched_row is not None:
                            col_resp_map = {
                                "base": matched_row.get(base_col, "—"),
                                "beta_0.1": matched_row.get(b01_col, "—"),
                                "beta_0.5": matched_row.get(b05_col, "—"),
                            }
                            st.caption(f"📄 Recorded experiment response (Prompt {matched_row['id']})")
                            st.write(str(col_resp_map[key]))
                        else:
                            st.warning("Custom prompt live generation requires PyTorch to finish building on the host. Select any of the lab prompts from the dropdown above to view responses immediately.")
