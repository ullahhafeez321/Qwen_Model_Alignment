"""
conclusion.py – Page 7: Evidence-based analysis and conclusion.
"""

import streamlit as st
from utils.data_loader import (
    load_training_metrics,
    load_saved_ratings,
)
import pandas as pd


CRITERIA = ["helpfulness", "relevance", "clarity", "instruction_following"]


def render():
    st.markdown("# 📝 Conclusion & Analysis")
    st.markdown(
        "Evidence-based answers to the assignment's analysis questions, "
        "drawn from the recorded experiment artefacts."
    )

    # ── Q1: Did DPO improve overall? ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Did DPO Improve Overall Response Quality?")

    ratings = load_saved_ratings()
    has_ratings = _has_real_ratings(ratings)

    if has_ratings:
        numeric = ratings.copy()
        for c in CRITERIA:
            numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            numeric.loc[numeric[c] == 0, c] = pd.NA
        avg = numeric.groupby("model")[CRITERIA].mean()
        avg["overall"] = avg.mean(axis=1)
        st.dataframe(avg.style.format("{:.2f}", na_rep="—"), use_container_width=True)
        st.markdown(
            "The table above shows average human evaluation scores. Compare "
            "the BASE row with DPO rows to determine if alignment improved "
            "response quality."
        )
    else:
        st.warning(
            "**Insufficient evidence.** No human evaluation ratings have been "
            "submitted yet. Based on the recorded outputs alone:\n\n"
            "- Many responses across all three configurations are **very similar "
            "or identical**, suggesting that the DPO training with 1 epoch on "
            "this 0.5B parameter model produced **minimal observable change** in "
            "generation behaviour for many prompts.\n\n"
            "- Some prompts show minor wording differences between configurations, "
            "but without human ratings it is **not possible to conclude** that "
            "these differences constitute an improvement.\n\n"
            "- **All three configurations** exhibit issues such as arithmetic errors, "
            "hallucinated terms, truncated responses, and irrelevant appended text. "
            "DPO training did not eliminate these problems.\n\n"
            "**Conclusion:** The available evidence does not establish that DPO "
            "improved overall response quality. Human evaluation is required."
        )

    # ── Q2: Which prompt types improved most? ────────────────────────────
    st.markdown("---")
    st.markdown("### 2. Which Prompt Types Improved Most?")

    if has_ratings:
        numeric = ratings.copy()
        for c in CRITERIA:
            numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
            numeric.loc[numeric[c] == 0, c] = pd.NA
        avg_task = numeric.groupby(["type", "model"])[CRITERIA].mean()
        avg_task["overall"] = avg_task.mean(axis=1)
        st.dataframe(avg_task.style.format("{:.2f}", na_rep="—"), use_container_width=True)
        st.markdown(
            "Compare BASE vs DPO rows within each task type. The task type "
            "with the largest positive difference in the DPO rows is the "
            "most improved category."
        )
    else:
        st.markdown(
            """
            From inspecting the recorded outputs:

            - **Explanation** and **Question Answering** prompts show the least
              variation between configurations. The base model already produces
              reasonable definitions and factual answers, and DPO does not
              visibly improve them.

            - **Advice** prompts (study plans) show some structural differences.
              The DPO-aligned models occasionally use more formatting (bold headers,
              numbered lists), which could indicate a marginal improvement in
              organisation—but this is a subjective assessment.

            - **Summarisation** prompts are problematic across all configurations.
              None of the models respect the requested bullet-point count.

            - **Reasoning** prompts contain arithmetic errors in all configurations.
              DPO does not fix factual or computational mistakes inherent in the
              base model.

            - **Professional writing** (email) responses are nearly identical across
              all three configurations.

            **Without human ratings, no task type can be identified as clearly improved.**
            """
        )

    # ── Q3: Did the base model perform better on any prompts? ────────────
    st.markdown("---")
    st.markdown("### 3. Did the Base Model Perform Better on Any Prompts?")

    st.markdown(
        """
        From the recorded outputs:

        - **Prompt 7 (HTTP vs HTTPS):** The beta 0.1 model expands HTTPS as
          "*Hijacked* Secure HyperText Transport Protocol" — a hallucinated
          and incorrect expansion. The base and beta 0.5 models use
          "Héldert Secure Hypertext Transfer ProtocoL", which is also incorrect
          but arguably less misleading than "Hijacked". In this case, neither
          DPO configuration clearly outperforms the base.

        - **Prompt 5 (time calculation):** The beta 0.1 model introduces
          additional incorrect numbers (adds 2 to 12, claims 14 hours available,
          then uses "1/3 × 20"). The base and beta 0.5 models also produce
          wrong Python code (using 18 hours instead of 12), but the beta 0.1
          version adds extra incorrect reasoning steps. The base may be slightly
          less wrong here.

        - For several prompts the outputs are identical, so the base does not
          perform worse but also does not perform better.

        **Overall:** There is no strong evidence that the base model is
        consistently better, but there are specific instances where a DPO
        configuration introduces new errors absent in the base model.
        """
    )

    # ── Q4: Patterns in chosen responses ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. What Patterns Appear in the Preference Dataset's Chosen Responses?")

    st.info(
        "⚠️ **Note:** The experiment artefacts do not include samples of the "
        "chosen/rejected pairs from the `trl-lib/ultrafeedback_binarized` "
        "dataset. This evidence would need to be generated by loading the "
        "dataset and inspecting example pairs.\n\n"
        "Based on the publicly documented characteristics of UltraFeedback:\n"
        "- Chosen responses tend to be **more detailed, well-structured, "
        "and instruction-following**.\n"
        "- Rejected responses may be **shorter, off-topic, or contain errors**.\n"
        "- The dataset covers a wide range of instruction types, including "
        "open-ended generation, reasoning, and coding.\n\n"
        "**To fully satisfy this requirement**, load the dataset and add "
        "example chosen/rejected pairs to the evidence."
    )

    # ── Q5: How did beta affect behaviour? ───────────────────────────────
    st.markdown("---")
    st.markdown("### 5. How Did the Beta Parameter Affect Behaviour?")

    try:
        m01 = load_training_metrics("0_1")
        m05 = load_training_metrics("0_5")

        st.markdown(
            f"""
            | Metric | β=0.1 | β=0.5 |
            |---|---|---|
            | **Final training loss** | {m01['train_loss']:.6f} | {m05['train_loss']:.6f} |
            | **Training runtime** | {m01['train_runtime']:.1f}s | {m05['train_runtime']:.1f}s |
            """
        )
    except Exception:
        pass

    st.markdown(
        """
        **Training-level effects:**
        - β=0.5 achieved a **lower final training loss** (0.6598 vs 0.6817),
          indicating stronger optimisation toward the preference signal. This
          is expected: a higher beta amplifies the magnitude of the preference
          gradient, allowing the loss to decrease more within the same number
          of steps.

        - β=0.5 showed **larger reward margins** in the training logs (up to
          0.209 vs 0.054 for β=0.1), meaning the model learned to assign higher
          implicit rewards to chosen responses relative to rejected ones.

        - β=0.5 also showed **higher gradient norms** (~470–715 vs ~104–144),
          reflecting the stronger gradient signal from the KL penalty.

        **Generation-level effects:**
        - Despite the training differences, the **generated responses are
          largely similar** between β=0.1 and β=0.5 for most prompts.

        - On some prompts (e.g. Prompt 1 Explanation, Prompt 5 Reasoning),
          β=0.5 produces output identical to the base model while β=0.1 shows
          minor variations. This is consistent with β=0.5's stronger KL
          constraint keeping the model closer to the base.

        - On other prompts (e.g. Prompt 7 HTTP/HTTPS), β=0.1 diverges more
          from the base (introducing different wording and errors) while β=0.5
          stays closer to the base output.

        **Summary:** The beta parameter had measurable effects on training
        dynamics but produced only subtle differences in generated text for
        this experiment. With a larger model or more training, the differences
        might become more pronounced.
        """
    )

    # ── Q6: What did DPO change? ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 6. What Did DPO Change Overall?")

    st.markdown(
        """
        **Summary of observed effects:**

        1. **Training successfully ran** for both beta values, producing valid
           LoRA adapters that can be loaded and used for generation.

        2. **Training loss decreased** from the initial ~0.693 (near random for
           binary preference classification) to 0.682 (β=0.1) and 0.660 (β=0.5),
           confirming that the model learned from the preference signal.

        3. **Preference accuracy improved** during training (from ~0.24 to ~0.66
           for β=0.1, and from ~0.13 to ~0.71 for β=0.5), showing the model
           progressively learned to distinguish chosen from rejected responses.

        4. **Generated outputs changed minimally** for most prompts. The LoRA
           adapters introduced small perturbations, but the 0.5B model's overall
           language generation capability was the dominant factor.

        5. **DPO did not fix fundamental model limitations:** factual errors,
           arithmetic mistakes, hallucinated terms, and formatting issues persist
           across all three configurations.

        6. **The experiment provides a valid baseline** for understanding DPO
           mechanics and the effect of beta, even though the alignment effect
           is modest on this small model with limited training.
        """
    )

    st.success(
        "**Key takeaway:** DPO training ran successfully and produced measurable "
        "changes in training metrics, but the generated text improvements on this "
        "0.5B model after 1 epoch are subtle. A larger model, more training data, "
        "or longer training would likely produce more visible alignment effects."
    )


def _has_real_ratings(df: pd.DataFrame) -> bool:
    """Check if the ratings DataFrame contains any actual submitted ratings."""
    if df.empty:
        return False
    for c in CRITERIA:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if (vals > 0).any():
                return True
    return False

