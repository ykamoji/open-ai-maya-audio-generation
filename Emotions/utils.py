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


def getModelAndTokenizer(MODEL_PATH, quantize, platform):
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

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct" if platform != 'Kaggle' else "/kaggle/input/llama-3-1-8b-instruct/transformers/1/1/Model",
        cache_dir=MODEL_PATH,
        quantization_config=bnb_config,
        device_map="balanced",
        torch_dtype=DTYPE,
        load_in_4bit=False,
        load_in_8bit=False
    )

    model.eval()

    model.generation_config.return_legacy_cache = True

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct" if platform != 'Kaggle' else "/kaggle/input/llama-3-1-8b-instruct/transformers/1/1/Tokenizer",
        use_fast=True,
        padding_side="left",
        cache_dir=MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def encode_no_bos(text, tokenizer, device):
    ids = tokenizer.encode(text, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
        ids = ids[1:]

    tensor = torch.tensor([ids], device=device)
    attn = torch.ones_like(tensor)
    return tensor, attn


def split_sentences(text: str):
    parts = sentence_regex.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def repeat_past_kv(past_key_values, batch_size):
    new_past = []
    for k, v in past_key_values:
        k_rep = k.repeat(batch_size, 1, 1, 1)
        v_rep = v.repeat(batch_size, 1, 1, 1)
        new_past.append((k_rep, v_rep))
    return tuple(new_past)


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
            for b in range(batch_size):
                seen = set(generated[b].tolist())
                for t in seen:
                    val = logits[b, t]
                    logits[b, t] = val / repetition_penalty if val > 0 else val * repetition_penalty

        # ----- 3. Top-k -----
        if top_k > 0:
            kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < kth_vals,
                torch.full_like(logits, -float('inf')),
                logits
            )

        # ----- 4. Top-p (nucleus) -----
        if top_p < 1.0:
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
