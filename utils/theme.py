"""
theme.py – Clean, responsive academic theme with light/dark mode compatibility.
"""

import streamlit as st

NAVY = "#1E3A5F"
NAVY_LIGHT = "#2C5282"
ACCENT = "#2563EB"


def inject_custom_css() -> None:
    """Inject universal CSS that looks clean and never breaks in dark or light mode."""
    st.markdown(
        """
        <style>
        /* ── Page spacing ────────────────────────────────── */
        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* ── Headers ─────────────────────────────────────── */
        h1 {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.25rem !important;
        }
        h2 {
            font-size: 1.35rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.4rem !important;
        }

        /* ── Universal Response Panel (Always readable) ──── */
        .response-box {
            background-color: var(--secondary-background-color, #F8FAFC);
            color: var(--text-color, #0F172A) !important;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
            font-size: 0.92rem;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 180px;
        }

        /* ── Badge Pills ─────────────────────────────────── */
        .badge {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }
        .badge-blue {
            background-color: #DBEAFE;
            color: #1E40AF !important;
        }
        .badge-green {
            background-color: #DCFCE7;
            color: #166534 !important;
        }
        .badge-orange {
            background-color: #FFEDD5;
            color: #9A3412 !important;
        }
        .badge-gray {
            background-color: #F1F5F9;
            color: #475569 !important;
        }

        /* ── Subtitle / description ──────────────────────── */
        .page-subtitle {
            font-size: 0.95rem;
            color: var(--text-color, #64748B);
            opacity: 0.85;
            margin-bottom: 1.25rem;
        }

        /* ── Footer ──────────────────────────────────────── */
        .app-footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            font-size: 0.82rem;
            color: #94A3B8;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            margin-top: 2.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = "blue") -> str:
    """Return a styled HTML badge pill."""
    return f'<span class="badge badge-{color}">{text}</span>'


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            <strong>DPO Alignment Demonstration</strong> · University LLM Lab 10 · Qwen2.5-0.5B
        </div>
        """,
        unsafe_allow_html=True,
    )
