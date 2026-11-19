"""
data_loader.py – Load and cache all experiment CSV / JSON artefacts.

Every public function uses @st.cache_data so the files are read only once
per Streamlit session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Data directory (relative to repo root) ──────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── CSV loaders ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_base_outputs() -> pd.DataFrame:
    """Base model outputs – columns: id, type, prompt, base_response."""
    return pd.read_csv(DATA_DIR / "base_outputs.csv")


@st.cache_data(show_spinner=False)
def load_beta_01_outputs() -> pd.DataFrame:
    """DPO beta 0.1 outputs – columns: id, type, prompt, beta_0.1_response."""
    return pd.read_csv(DATA_DIR / "beta_0_1_outputs.csv")


@st.cache_data(show_spinner=False)
def load_beta_05_outputs() -> pd.DataFrame:
    """DPO beta 0.5 outputs – columns: id, type, prompt, beta_0.5_response."""
    return pd.read_csv(DATA_DIR / "beta_0_5_outputs.csv")


@st.cache_data(show_spinner=False)
def load_side_by_side() -> pd.DataFrame:
    """Combined comparison – columns include all three responses."""
    return pd.read_csv(DATA_DIR / "side_by_side_comparison.csv")


@st.cache_data(show_spinner=False)
def load_training_summary() -> pd.DataFrame:
    """training_metrics_summary.csv (beta, train_runtime, train_loss …)."""
    return pd.read_csv(DATA_DIR / "training_metrics_summary.csv")


@st.cache_data(show_spinner=False)
def load_human_ratings_template() -> pd.DataFrame:
    """human_ratings_template.csv – blank template."""
    return pd.read_csv(DATA_DIR / "human_ratings_template.csv")


# ── JSON loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_training_metrics(beta: str) -> dict[str, Any]:
    """Return the per-beta training_metrics JSON as a dict."""
    fname = f"training_metrics_beta_{beta}.json"
    with open(DATA_DIR / fname) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_adapter_config(beta: str) -> dict[str, Any]:
    """Return the adapter_config.json for the given beta (e.g. '0_1')."""
    fname = f"adapter_config_beta_{beta}.json"
    with open(DATA_DIR / fname) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_trainer_state(beta: str) -> dict[str, Any]:
    """Return the trainer_state.json for the given beta."""
    fname = f"trainer_state_beta_{beta}.json"
    with open(DATA_DIR / fname) as f:
        return json.load(f)


# ── Convenience: merged comparison dataframe ────────────────────────────
@st.cache_data(show_spinner=False)
def load_merged_comparison() -> pd.DataFrame:
    """
    Merge base, beta-0.1, and beta-0.5 outputs on (id, type, prompt).
    Falls back to side_by_side_comparison.csv if merge fails.
    """
    try:
        base = load_base_outputs()
        b01 = load_beta_01_outputs()
        b05 = load_beta_05_outputs()
        merged = base.merge(b01, on=["id", "type", "prompt"], how="outer")
        merged = merged.merge(b05, on=["id", "type", "prompt"], how="outer")
        return merged
    except Exception:
        return load_side_by_side()


# ── Human ratings persistence ───────────────────────────────────────────
RATINGS_FILE = DATA_DIR / "human_ratings.csv"


def load_saved_ratings() -> pd.DataFrame:
    """Load user-submitted ratings, or return an empty DataFrame."""
    if RATINGS_FILE.exists():
        df = pd.read_csv(RATINGS_FILE)
        if not df.empty:
            return df
    return pd.DataFrame(columns=[
        "id", "type", "model", "helpfulness", "relevance",
        "clarity", "instruction_following", "notes",
    ])


def save_ratings(df: pd.DataFrame) -> None:
    """Persist ratings to CSV."""
    df.to_csv(RATINGS_FILE, index=False)

