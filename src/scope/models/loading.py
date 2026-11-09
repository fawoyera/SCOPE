"""Model-agnostic loading for any Hugging Face causal LM listed in configs/models/.

Adding a model = adding a YAML file with its ``hf_id``; nothing here is model specific
except sensible defaults (pad token, attention backend, decoder-layer discovery).
"""
from __future__ import annotations

import inspect
from typing import List, Optional

import torch
from torch import nn

from ..config import ModelConfig
from ..utils.common import log

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def n_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def input_device(model) -> torch.device:
    """Device of the input embeddings (first pipeline stage under device_map=auto)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _attn_impl(requested: str) -> str:
    if requested != "auto":
        return requested
    if n_gpus() == 0:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        ver = tuple(int(x) for x in flash_attn.__version__.split(".")[:3])
        if ver >= (2, 1, 0):
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


def load_tokenizer(cfg: ModelConfig, path: Optional[str] = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path or cfg.hf_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"      # training batches; generation pads left explicitly
    tok.truncation_side = "left"    # never truncate the end of a prompt
    return tok


def load_model(cfg: ModelConfig, path: Optional[str] = None, trainable: bool = True):
    from transformers import AutoModelForCausalLM
    attn = _attn_impl(cfg.attn_implementation)
    dtype = DTYPES[cfg.dtype] if n_gpus() else torch.float32
    import transformers
    from packaging.version import Version
    dkey = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kw = {dkey: dtype, "attn_implementation": attn, "trust_remote_code": cfg.trust_remote_code}
    multi = n_gpus() >= 2 and cfg.device_map == "auto"
    if multi:
        kw["device_map"] = "auto"
    src = path or cfg.hf_id
    try:
        model = AutoModelForCausalLM.from_pretrained(src, **kw)
    except (ValueError, ImportError) as e:   # e.g. attention backend unsupported by this arch
        log(f"Falling back to sdpa attention ({e})")
        kw["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(src, **kw)
    if not multi:
        model.to("cuda:0" if n_gpus() else "cpu")
    if cfg.gradient_checkpointing and trainable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if not trainable:
        model.eval().requires_grad_(False)
    log(f"Loaded {src} | attn={kw['attn_implementation']} dtype={dtype} "
        f"| {'pipeline over %d GPUs' % n_gpus() if multi else input_device(model)}")
    return model


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Largest ModuleList of transformer blocks (model.layers, transformer.h, ...)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and name.split(".")[-1] in ("layers", "h", "blocks", "layer"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("Could not locate decoder layers")
    return best


def shared_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Parameters used by GradNorm to measure per-task gradient norms (last block)."""
    return [p for p in decoder_layers(model)[-1].parameters() if p.requires_grad]


def supports_logits_to_keep(model) -> bool:
    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
