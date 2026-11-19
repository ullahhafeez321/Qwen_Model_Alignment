"""
training_analysis.py – Page 5: Training metrics visualisation.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_training_metrics,
    load_training_summary,
    load_trainer_state,
)
from utils.theme import metric_card


def render():
    st.markdown("# 📈 Training Analysis")
    st.markdown(
        "Training metrics recorded during the DPO fine-tuning runs for both "
        "beta configurations."
    )

    # ── Summary cards ────────────────────────────────────────────────────
    st.markdown("### Training Summary")
    try:
        m01 = load_training_metrics("0_1")
        m05 = load_training_metrics("0_5")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("β=0.1 Loss", f"{m01['train_loss']:.4f}"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("β=0.5 Loss", f"{m05['train_loss']:.4f}"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("β=0.1 Runtime", f"{m01['train_runtime']:.1f}s"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("β=0.5 Runtime", f"{m05['train_runtime']:.1f}s"), unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Could not load training metrics: {exc}")
        return

    st.markdown("")
    st.warning(
        "⚠️ **Interpretation note:** A lower training loss means the model's "
        "predictions moved closer to the preference-dataset signal during training. "
        "It does **not** by itself prove that the model's responses are better-aligned "
        "or higher-quality. Evaluation on held-out prompts and human judgement are "
        "needed for that conclusion."
    )

    # ── Training loss curve ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Training Loss Over Steps")

    try:
        state_01 = load_trainer_state("0_1")
        state_05 = load_trainer_state("0_5")

        log_01 = [
            entry for entry in state_01["log_history"]
            if "loss" in entry and "step" in entry
        ]
        log_05 = [
            entry for entry in state_05["log_history"]
            if "loss" in entry and "step" in entry
        ]

        if log_01 and log_05:
            df_01 = pd.DataFrame(log_01)[["step", "loss"]].rename(columns={"loss": "β=0.1 Loss"})
            df_05 = pd.DataFrame(log_05)[["step", "loss"]].rename(columns={"loss": "β=0.5 Loss"})
            merged = df_01.merge(df_05, on="step", how="outer").sort_values("step")

            # Use plotly for interactive chart
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=merged["step"], y=merged["β=0.1 Loss"],
                    mode="lines+markers", name="β=0.1",
                    line=dict(color="#3182CE", width=2),
                    marker=dict(size=5),
                ))
                fig.add_trace(go.Scatter(
                    x=merged["step"], y=merged["β=0.5 Loss"],
                    mode="lines+markers", name="β=0.5",
                    line=dict(color="#DD6B20", width=2),
                    marker=dict(size=5),
                ))
                fig.update_layout(
                    xaxis_title="Training Step",
                    yaxis_title="Loss",
                    template="plotly_white",
                    height=420,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    margin=dict(t=30),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(merged.set_index("step"))

            st.caption(
                "Both runs complete 128 steps (1 epoch). β=0.5 shows a steeper "
                "loss decrease, reaching a lower final training loss (0.6598 vs "
                "0.6817). This is expected because a higher beta amplifies the "
                "gradient signal from preference margins."
            )
        else:
            st.info("No step-level loss data available in trainer state.")
    except Exception as exc:
        st.warning(f"Could not plot training loss curve: {exc}")

    # ── Rewards / accuracies ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### DPO Rewards & Preference Accuracy")

    try:
        for beta_label, beta_key, color in [
            ("β=0.1", "0_1", "#3182CE"),
            ("β=0.5", "0_5", "#DD6B20"),
        ]:
            state = load_trainer_state(beta_key)
            entries = [
                e for e in state["log_history"]
                if "rewards/accuracies" in e and "step" in e
            ]
            if entries:
                df_r = pd.DataFrame(entries)

                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[
                            f"Preference Accuracy ({beta_label})",
                            f"Reward Margins ({beta_label})",
                        ],
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_r["step"], y=df_r["rewards/accuracies"],
                            mode="lines+markers", name="Accuracy",
                            line=dict(color=color),
                        ),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_r["step"], y=df_r["rewards/margins"],
                            mode="lines+markers", name="Margins",
                            line=dict(color=color, dash="dash"),
                        ),
                        row=1, col=2,
                    )
                    fig.update_layout(
                        height=320, template="plotly_white",
                        showlegend=False, margin=dict(t=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.dataframe(df_r[["step", "rewards/accuracies", "rewards/margins"]])
    except Exception as exc:
        st.warning(f"Could not plot reward data: {exc}")

    # ── Detailed log table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Detailed Training Log")

    tab1, tab2 = st.tabs(["β=0.1 Log", "β=0.5 Log"])
    for tab, beta_key, label in [(tab1, "0_1", "β=0.1"), (tab2, "0_5", "β=0.5")]:
        with tab:
            try:
                state = load_trainer_state(beta_key)
                log_entries = [
                    e for e in state["log_history"]
                    if "step" in e and "loss" in e
                ]
                if log_entries:
                    df_log = pd.DataFrame(log_entries)
                    display_cols = [
                        c for c in [
                            "step", "epoch", "loss", "grad_norm",
                            "learning_rate", "rewards/accuracies",
                            "rewards/chosen", "rewards/rejected",
                            "rewards/margins",
                        ]
                        if c in df_log.columns
                    ]
                    st.dataframe(
                        df_log[display_cols].style.format(precision=6),
                        use_container_width=True,
                        height=400,
                    )
                else:
                    st.info(f"No step-level log entries for {label}.")
            except Exception as exc:
                st.error(f"Could not load trainer state for {label}: {exc}")

    # ── Runtime comparison ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Runtime Comparison")
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                x=["β=0.1", "β=0.5"],
                y=[m01["train_runtime"], m05["train_runtime"]],
                marker_color=["#3182CE", "#DD6B20"],
                text=[f"{m01['train_runtime']:.1f}s", f"{m05['train_runtime']:.1f}s"],
                textposition="auto",
            )
        ])
        fig.update_layout(
            yaxis_title="Runtime (seconds)",
            template="plotly_white",
            height=350,
            margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Both runs took approximately the same time (~384–386 seconds), "
            "which is expected since the beta parameter affects the loss "
            "computation but not the number of training steps or batch size."
        )
    except ImportError:
        st.markdown(
            f"- β=0.1 runtime: **{m01['train_runtime']:.1f}s**\n"
            f"- β=0.5 runtime: **{m05['train_runtime']:.1f}s**"
        )

