"""
training_analysis.py – Training metrics and loss curves.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_training_metrics,
    load_training_summary,
    load_trainer_state,
)


def render():
    st.markdown("## 📈 Training Metrics & Optimization Dynamics")
    st.caption("Empirical training curves and logs recorded over 128 optimization steps (1.0 epoch) on UltraFeedback.")

    # ── Summary metrics ──────────────────────────────────────────────────
    try:
        m01 = load_training_metrics("0_1")
        m05 = load_training_metrics("0_5")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("β=0.1 Final Loss", f"{m01['train_loss']:.4f}", "-0.012")
        c2.metric("β=0.5 Final Loss", f"{m05['train_loss']:.4f}", "-0.033")
        c3.metric("β=0.1 Runtime", f"{m01['train_runtime']:.1f}s")
        c4.metric("β=0.5 Runtime", f"{m05['train_runtime']:.1f}s")
    except Exception as exc:
        st.error(f"Could not load metrics: {exc}")
        return

    st.markdown("---")

    # ── Step-level Loss Curves ───────────────────────────────────────────
    st.markdown("### Training Loss Progression (128 Steps)")
    try:
        state_01 = load_trainer_state("0_1")
        state_05 = load_trainer_state("0_5")

        log_01 = [e for e in state_01["log_history"] if "loss" in e and "step" in e]
        log_05 = [e for e in state_05["log_history"] if "loss" in e and "step" in e]

        if log_01 and log_05:
            df_01 = pd.DataFrame(log_01)[["step", "loss"]].rename(columns={"loss": "β=0.1 Loss"})
            df_05 = pd.DataFrame(log_05)[["step", "loss"]].rename(columns={"loss": "β=0.5 Loss"})
            merged = df_01.merge(df_05, on="step", how="outer").sort_values("step")

            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["step"], y=merged["β=0.1 Loss"],
                mode="lines+markers", name="DPO β=0.1",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=merged["step"], y=merged["β=0.5 Loss"],
                mode="lines+markers", name="DPO β=0.5",
                line=dict(color="#EA580C", width=2.5),
                marker=dict(size=6),
            ))
            fig.update_layout(
                xaxis_title="Optimization Step",
                yaxis_title="DPO Loss",
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("📌 **Takeaway:** β=0.5 enforces a stronger preference margin gradient, resulting in lower loss (0.6598 vs 0.6817). Note that lower training loss does not automatically equate to higher generative quality.")
    except Exception as exc:
        st.warning(f"Could not render loss curve: {exc}")

    # ── Preference Accuracy & Reward Margins ─────────────────────────────
    st.markdown("---")
    st.markdown("### Preference Accuracy & Reward Margins")

    try:
        entries_01 = [e for e in state_01["log_history"] if "rewards/accuracies" in e and "step" in e]
        entries_05 = [e for e in state_05["log_history"] if "rewards/accuracies" in e and "step" in e]

        if entries_01 and entries_05:
            df_r01 = pd.DataFrame(entries_01)
            df_r05 = pd.DataFrame(entries_05)

            col_acc, col_marg = st.columns(2)

            with col_acc:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=df_r01["step"], y=df_r01["rewards/accuracies"],
                    mode="lines+markers", name="β=0.1 Accuracy",
                    line=dict(color="#2563EB"),
                ))
                fig_acc.add_trace(go.Scatter(
                    x=df_r05["step"], y=df_r05["rewards/accuracies"],
                    mode="lines+markers", name="β=0.5 Accuracy",
                    line=dict(color="#EA580C"),
                ))
                fig_acc.update_layout(
                    title="Preference Classification Accuracy",
                    xaxis_title="Step",
                    yaxis_title="Accuracy (0 to 1)",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_marg:
                fig_marg = go.Figure()
                fig_marg.add_trace(go.Scatter(
                    x=df_r01["step"], y=df_r01["rewards/margins"],
                    mode="lines+markers", name="β=0.1 Margin",
                    line=dict(color="#2563EB", dash="dot"),
                ))
                fig_marg.add_trace(go.Scatter(
                    x=df_r05["step"], y=df_r05["rewards/margins"],
                    mode="lines+markers", name="β=0.5 Margin",
                    line=dict(color="#EA580C", dash="dot"),
                ))
                fig_marg.update_layout(
                    title="Implicit Reward Margins (Chosen - Rejected)",
                    xaxis_title="Step",
                    yaxis_title="Margin",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_marg, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render accuracy/margin charts: {exc}")
