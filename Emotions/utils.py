import torch
import re
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sentence_regex = re.compile(
    r'''(?x)
    (?<!\w\.\w.)          
    (?<![A-Z][a-z]\.)     
    (?<=\.|\?|!)          
    \s+                   
    '''
)

TONES = [
    "[ANGRY]", "[EXCITED]", "[SARCASTIC]", "[CURIOUS]", "[SING]", "[APPALLED]", "[MISCHIEVOUS]", "[DISAPPOINTED]"
]

SOUNDS = [
    "[LAUGH]", "[LAUGH_HARDER]", "[SIGH]", "[CHUCKLE]", "[GASP]", "[CRY]", "[SCREAM]", "[WHISPER]", "[SNORT]",
    "[EXHALE]", "[GULP]", "[GIGGLE]"
]

TTS_TAGS = TONES + SOUNDS

TTS_TAGS_STR = ", ".join(TTS_TAGS)


def getDevice():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    return device


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def getModelAndTokenizer(Args):
    MODEL_NAME = Args.Emotions.ModelName.__dict__[Args.Platform]
    CACHE_PATH = Args.Emotions.CachePath.__dict__[Args.Platform]

    quantize, platform = Args.Emotions.Quantize, Args.Platform


    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16

    bnb_config = None
    if quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_PATH,
        use_fast=True,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_PATH,
        quantization_config=bnb_config,
        device_map="cpu",
        torch_dtype=DTYPE,
        load_in_4bit=False,
        load_in_8bit=False
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def split_sentences(text: str):
    parts = sentence_regex.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def repeat_past_kv(past_key_values, batch_size):
    new_past = []
    for k, v in past_key_values:
        k_exp = k.expand(batch_size, -1, -1, -1).contiguous()
        v_exp = v.expand(batch_size, -1, -1, -1).contiguous()
        new_past.append((k_exp, v_exp))
    return tuple(new_past)


def slice_prefix_kv(prefix_kv_big, batch_size):
    sliced = []
    for k_big, v_big in prefix_kv_big:
        k_small = k_big[:batch_size]
        v_small = v_big[:batch_size]
        sliced.append((k_small, v_small))
    return tuple(sliced)


@torch.inference_mode()
def fast_generate(
        model,
        dynamic_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id=None,
        past_key_values=None,
):
    """
    Greedy decoding with optional prefilled KV cache (prefix).

    - dynamic_ids: (batch, dynamic_len) – ONLY the suffix tokens (no static prefix).
    - attention_mask: (batch, prefix_len + dynamic_len) when past_key_values is provided.
    - past_key_values:
        * None → full prefill from dynamic_ids only.
        * Not None → treated as prefix KV; dynamic_ids are appended after it.
    """

    device = next(model.parameters()).device
    model.eval()

    dynamic_ids = dynamic_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Normalize EOS
    if eos_token_id is None:
        eos_tensor = None
    else:
        eos_tensor = torch.tensor([eos_token_id], device=device)

    batch_size = dynamic_ids.size(0)

    # Initial forward pass (prefix KV + dynamic tokens)
    out = model(
        input_ids=dynamic_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = out.logits[:, -1, :]  # (batch, vocab)
    past = out.past_key_values  # includes prefix + dynamic

    # We'll decode only from the dynamic part onward
    generated = dynamic_ids.clone()

    # Track finished if EOS is defined
    if eos_tensor is not None:
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        finished = None

    cur_attention_mask = attention_mask

    for _ in range(max_new_tokens):
        next_tok = torch.argmax(logits, dim=-1)  # (batch,)

        if eos_tensor is not None:
            is_eos = (next_tok == eos_tensor[0])
            finished |= is_eos

            generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)

            if finished.all():
                break
        else:
            generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)

        # extend attention mask (prefix_len + dynamic_len + t)
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones_like(next_tok).unsqueeze(-1)],
            dim=-1,
        )

        out = model(
            input_ids=next_tok.unsqueeze(-1),  # (batch, 1)
            attention_mask=cur_attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past = out.past_key_values

    return generated


@torch.inference_mode()
def fast_generate_sampling(
        model,
        dynamic_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        max_new_tokens: int = 128,
        eos_token_id=None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
):
    """
    Trimmed KV-cached HF-compatible sampler.
    Supports ONLY:
        - temperature
        - top_k
        - top_p
        - repetition_penalty
        - eos_token_id
        - max_new_tokens

    Matches HuggingFace generate(do_sample=True) for these params.
    Optimized for speed.
    """

    device = next(model.parameters()).device
    model.eval()

    dynamic_ids = dynamic_ids.to(device)
    attention_mask = attention_mask.to(device)

    eos_tensor = None
    if eos_token_id is not None:
        eos_tensor = torch.tensor([eos_token_id], device=device)

    batch_size = dynamic_ids.size(0)

    # ---- Prefill dynamic ----
    out = model(
        input_ids=dynamic_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True
    )

    logits = out.logits[:, -1, :]
    past = out.past_key_values

    generated = dynamic_ids.clone()

    # Track finished sequences
    if eos_tensor is not None:
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        finished = None

    cur_mask = attention_mask

    # -----------------------------
    #       Decode Loop
    # -----------------------------
    for step in range(max_new_tokens):

        # ----- 1. Temperature scaling -----
        if temperature != 1.0:
            logits = logits / temperature

        # ----- 2. Repetition penalty -----
        if repetition_penalty != 1.0:
            before = logits.clone()
            for b in range(batch_size):
                seen = set(generated[b].tolist())
                for t in seen:
                    val = logits[b, t]
                    logits[b, t] = val / repetition_penalty if val > 0 else val * repetition_penalty

            # per-filter fallback
            if torch.isnan(logits).any() or torch.isinf(logits).all():
                logits = before

        # ----- 3. Top-k -----
        if top_k > 0:
            before = logits.clone()
            kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < kth_vals,
                torch.full_like(logits, -float('inf')),
                logits
            )
            # per-filter fallback
            if torch.isinf(logits).all() or torch.isnan(logits).any():
                logits = before

        # ----- 4. Top-p (nucleus) -----
        if top_p < 1.0:
            before = logits.clone()

            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            mask = cumulative_probs > top_p
            mask[:, 1:] = mask[:, :-1].clone()

            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, -float('inf')), sorted_logits)

            # Unsort
            original = torch.empty_like(sorted_logits)
            original.scatter_(1, sorted_idx, sorted_logits)
            logits = original

            # per-filter fallback
            if torch.isinf(logits).all() or torch.isnan(logits).any():
                logits = before

        # ----- 5. Sample next token -----
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1).squeeze(-1)

        # ----- 6. Handle EOS -----
        if eos_tensor is not None:
            eos_reached = (next_tok == eos_tensor[0])

            # Suppress EOS until min_new_tokens reached
            if step < min_new_tokens:
                # replace EOS with highest non-EOS token
                for b in range(batch_size):
                    if eos_reached[b]:
                        # pick second-best token that is not EOS
                        _, top2 = torch.topk(logits[b], 2)
                        replacement = top2[1] if top2[0] == eos_tensor[0] else top2[0]
                        next_tok[b] = replacement

            else:
                # after min_new_tokens, allow EOS
                finished |= eos_reached

        # Append sampled token
        generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)

        # Stop if all sequences finished
        if finished is not None and finished.all():
            break

        # ----- 7. Update attention mask -----
        cur_mask = torch.cat(
            [cur_mask, torch.ones_like(next_tok).unsqueeze(-1)],
            dim=-1
        )

        # ----- 8. Next forward-step using KV cache -----
        out = model(
            input_ids=next_tok.unsqueeze(-1),
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True
        )

        logits = out.logits[:, -1, :]
        past = out.past_key_values

    return generated
