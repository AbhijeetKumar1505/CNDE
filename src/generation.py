"""Local LLM loading, prompt construction and answer generation.  (Owner: Member 2)

Contract
--------
load_llm(model_id, compute_dtype) -> (model, tokenizer)     # 4-bit NF4, double quant, FP16 compute
smoke_test(model, tokenizer) -> dict                        # finite-logits check, VRAM, tokens/s
build_messages(question, evidence, variant) -> list[dict]   # chat messages; evidence labelled [S1]..[Sn]
generate_answer(question, evidence, model, tokenizer, variant, max_new_tokens) -> dict
    {"answer": str, "cited_ids": list[str], "abstained": bool,
     "prompt_tokens": int, "new_tokens": int, "latency_s": float}
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def gpu_memory() -> dict:
    """Current/peak VRAM in GB as seen by PyTorch, plus device-wide usage from the driver."""
    free, total = torch.cuda.mem_get_info()
    return {
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "vram_device_used_gb": round((total - free) / 1e9, 2),
        "vram_total_gb": round(total / 1e9, 2),
    }


def load_llm(model_id, compute_dtype=torch.float16):
    """Load a causal LM in 4-bit NF4 with double quantization on a single GPU.

    - FP16 compute: the T4 (Turing) has no native BF16.
    - torch_dtype is passed explicitly: transformers v5 defaults to dtype="auto", which would pick the
      checkpoint's bfloat16. torch_dtype is still honoured by v5 (mapped onto dtype) and by v4.51+.
    - device_map={"": 0} keeps every layer on the GPU and fails loudly on OOM instead of silently
      offloading to CPU (which "auto" may do).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible. In Colab: Runtime > Change runtime type > T4 GPU.")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )
    model.eval()
    print(f"Loaded {model_id} in {time.perf_counter() - t0:.1f}s | {gpu_memory()}")
    return model, tokenizer


@torch.inference_mode()
def smoke_test(model, tokenizer, max_new_tokens=64) -> dict:
    """Detect FP16 overflow (non-finite logits) and measure greedy decode speed and VRAM."""
    messages = [{"role": "user", "content": "In one sentence, what is ultrasonic testing in NDT?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    logits = model(**inputs).logits
    finite = bool(torch.isfinite(logits).all())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,  # silence sampling-flag warnings under greedy decoding
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "logits_finite": finite,
        "new_tokens": int(new_ids.numel()),
        "tokens_per_s": round(new_ids.numel() / seconds, 1),
        **gpu_memory(),
    }


def build_messages(question, evidence, variant="v2"):
    raise NotImplementedError("Milestone 3")


def generate_answer(question, evidence, model, tokenizer, variant="v2", max_new_tokens=512):
    raise NotImplementedError("Milestone 3")
