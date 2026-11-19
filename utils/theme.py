"""
theme.py – Shared CSS and UI helper functions for the DPO Lab 10 dashboard.
"""

import streamlit as st

NAVY = "#1E3A5F"
NAVY_LIGHT = "#2C5282"
ACCENT = "#3182CE"
BG_CARD = "#F7FAFC"
BORDER = "#CBD5E0"
TEXT_MUTED = "#718096"


def inject_custom_css() -> None:
    """Inject project-wide CSS into the Streamlit page."""
    st.markdown(
        f"""
        <style>
        /* ── Global ─────────────────────────────────────── */
        .main .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
        }}
        h1, h2, h3 {{
            color: {NAVY};
        }}
        /* ── Metric cards ───────────────────────────────── */
        .metric-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 0.85rem;
            color: {TEXT_MUTED};
            margin-bottom: 0.3rem;
        }}
        .metric-card .value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {NAVY};
        }}
        /* ── Response panels ────────────────────────────── */
        .response-panel {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            font-size: 0.92rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .response-panel-header {{
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}
        /* ── Badge pills ────────────────────────────────── */
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 99px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }}
        .badge-blue {{
            background: #EBF4FF;
            color: {ACCENT};
        }}
        .badge-green {{
            background: #F0FFF4;
            color: #38A169;
        }}
        .badge-orange {{
            background: #FFFAF0;
            color: #DD6B20;
        }}
        .badge-gray {{
            background: #EDF2F7;
            color: #4A5568;
        }}
        /* ── Footer ─────────────────────────────────────── */
        .footer {{
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            border-top: 1px solid {BORDER};
            margin-top: 3rem;
        }}
        /* ── Checklist ──────────────────────────────────── */
        .checklist-item {{
            padding: 0.5rem 0;
            border-bottom: 1px solid #EDF2F7;
        }}
        /* ── Tabs override for uniformity ────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 20px;
            border-radius: 6px 6px 0 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    """Return HTML for a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <strong>DPO Alignment Experiment Dashboard</strong><br>
            University LLM Lab 10 · Qwen/Qwen2.5-0.5B · Direct Preference Optimization<br>
            Built with Streamlit · Data sourced from recorded experiment artefacts
        </div>
        """,
        unsafe_allow_html=True,
    )

