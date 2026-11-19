"""
model_loader.py – Memory-safe, cached multi-adapter loading for Qwen2.5-0.5B.

Architecture:
- Loads Qwen/Qwen2.5-0.5B ONCE in float32 / float16.
- Wraps in PeftModel and loads both LoRA adapters (beta_0_1 and beta_0_5)
  onto the same base model to eliminate redundant VRAM/RAM copies.
- Switches between Base (via disable_adapter()), DPO beta 0.1, and DPO beta 0.5.
- Gracefully handles missing dependencies or hardware limitations.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import streamlit as st

HF_REPO_ID = os.environ.get("HF_REPO_ID", "ullahhafeez321/qwen-dpo-lab10")
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"

ADAPTER_SUBFOLDERS = {
    "beta_0.1": "Lab10_DPO_experiment_evidence/adapter_beta_0_1",
    "beta_0.5": "Lab10_DPO_experiment_evidence/adapter_beta_0_5",
}


def _get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return None


HF_TOKEN = _get_hf_token()


# ── Availability probe ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def inference_available() -> tuple[bool, str]:
    """Return (available, reason) for live inference capability."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        return False, f"Missing dependency: {exc.name}"
    return True, ""


def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Unified Model & Tokenizer Singleton ──────────────────────────────────
@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B base model & DPO adapters...")
def load_unified_peft_model():
    """
    Load base model once and attach both DPO adapters into a single PeftModel.
    Returns (peft_model, tokenizer, device).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import snapshot_download

    device = _get_device()
    dtype = torch.float16 if device != "cpu" else torch.float32

    # 1. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Base model setup
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    # 3. Download & attach beta_0.1 adapter
    dir_01 = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{ADAPTER_SUBFOLDERS['beta_0.1']}/*"],
        token=HF_TOKEN,
    )
    path_01 = os.path.join(dir_01, ADAPTER_SUBFOLDERS["beta_0.1"])
    peft_model = PeftModel.from_pretrained(base_model, path_01, adapter_name="beta_0_1")

    # 4. Download & attach beta_0.5 adapter to the same PeftModel
    dir_05 = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{ADAPTER_SUBFOLDERS['beta_0.5']}/*"],
        token=HF_TOKEN,
    )
    path_05 = os.path.join(dir_05, ADAPTER_SUBFOLDERS["beta_0.5"])
    peft_model.load_adapter(path_05, adapter_name="beta_0_5")

    peft_model.eval()
    return peft_model, tokenizer, device


# ── Generation ──────────────────────────────────────────────────────────
def generate_response(
    prompt: str,
    model_key: str = "base",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> Tuple[str, float, int]:
    """
    Generate a response for *prompt* using the selected model variant.

    model_key: 'base', 'beta_0.1', or 'beta_0.5'
    Returns: (decoded_response, elapsed_time_seconds, token_count)
    """
    import torch

    model, tokenizer, device = load_unified_peft_model()

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

    # Tokenizer end-of-turn IDs: Qwen <|endoftext|> is 151643, <|im_end|> is 151645
    eos_token_id = [tokenizer.eos_token_id]
    if 151645 not in eos_token_id:
        eos_token_id.append(151645)

    start_time = time.time()

    with torch.inference_mode():
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            ))
        else:
            gen_kwargs["do_sample"] = False

        if model_key == "base":
            with model.disable_adapter():
                outputs = model.generate(**gen_kwargs)
        elif model_key in ("beta_0.1", "beta_0_1"):
            model.set_adapter("beta_0_1")
            outputs = model.generate(**gen_kwargs)
        elif model_key in ("beta_0.5", "beta_0_5"):
            model.set_adapter("beta_0_5")
            outputs = model.generate(**gen_kwargs)
        else:
            with model.disable_adapter():
                outputs = model.generate(**gen_kwargs)

    elapsed = time.time() - start_time
    new_tokens = outputs[0][prompt_length:]
    token_count = len(new_tokens)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response, elapsed, token_count
