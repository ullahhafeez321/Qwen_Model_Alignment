"""
model_loader.py – Lazy, memory-safe loading of Qwen2.5-0.5B + LoRA adapters.

Design decisions:
- Load the base model ONCE per process and cache it in st.session_state.
- Apply a PEFT adapter on top; unload before switching to another adapter.
- Fall back gracefully when torch / transformers / peft are absent or when
  the host has insufficient memory.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

# ── HuggingFace repo layout ─────────────────────────────────────────────
HF_REPO_ID = os.environ.get(
    "HF_REPO_ID", "ullahhafeez321/qwen-dpo-lab10"
)
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"

ADAPTER_SUBFOLDERS = {
    "beta_0.1": "Lab10_DPO_experiment_evidence/adapter_beta_0_1",
    "beta_0.5": "Lab10_DPO_experiment_evidence/adapter_beta_0_5",
}


# ── Availability probe ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def inference_available() -> tuple[bool, str]:
    """Return (available, reason) for live inference capability."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from peft import PeftModel  # noqa: F401
    except ImportError as exc:
        return False, f"Missing dependency: {exc.name}"
    return True, ""


# ── Base model + tokenizer (session-cached) ─────────────────────────────
def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner="Loading base model …")
def get_base_model_and_tokenizer():
    """Load and cache the base Qwen2.5-0.5B model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _get_device()
    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource(show_spinner="Loading DPO adapter …")
def get_adapter_model(adapter_key: str):
    """Return PeftModel for the requested adapter (e.g. 'beta_0.1')."""
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    base_model, tokenizer, device = get_base_model_and_tokenizer()

    subfolder = ADAPTER_SUBFOLDERS[adapter_key]

    # Download adapter files
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{subfolder}/*"],
        token=HF_TOKEN,
    )
    adapter_path = os.path.join(local_dir, subfolder)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer, device


# ── Generation ──────────────────────────────────────────────────────────
def generate_response(
    prompt: str,
    model_key: str = "base",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response for *prompt* using the selected model variant.

    model_key: 'base', 'beta_0.1', or 'beta_0.5'
    Returns the decoded text of **only** the newly generated tokens.
    """
    import torch

    if model_key == "base":
        model, tokenizer, device = get_base_model_and_tokenizer()
    else:
        model, tokenizer, device = get_adapter_model(model_key)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][prompt_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()

