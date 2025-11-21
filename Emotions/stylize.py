import torch
import inspect
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer, fast_generate, repeat_past_kv
from utils import updateCache
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


SYSTEM_PROMPT = inspect.cleandoc("""
You are a professional editor preparing manuscripts for audiobook narration. Your task is to refine the paragraph to improve clarity, rhythm, 
and spoken-flow readability while maintaining all story details and staying true to the author’s voice.

Very important Guidelines:
- Preserve the emotional intent, atmosphere, and style of the writing.
- Maintain the pacing and dramatic beats of the scene.
- Keep character voice, tone, and attitude unchanged.
- Correct grammar, sentence structure, punctuation, and word only where it 
  enhances clarity or improves how the text sounds when read aloud.
- If the author uses vague or informal language (e.g., "stuff","things","somewhere"), keep it vague. 
  Do not specify or replace it with more precise terms.
- Strengthen readability and natural rhythm for voice actors without altering meaning.
- Avoid introducing new imagery, metaphors, or descriptive elements.
- Avoid removing key details or shortening passages in ways that alter the feel.
- Keep the writing consistent from sentence to sentence and across scenes.
- Do not over-polish or make the prose sound generic or mechanical.

Return only the edited paragraph, clean and ready for audiobook production.
""")


def build_system_prefix_cache(model, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        add_generation_prompt=False,
        tokenize=False,
    )

    enc = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            use_cache=True
        )

    static_ids = enc["input_ids"]          # (1, P)
    static_mask = enc["attention_mask"]    # (1, P)
    static_past = tuple((k.detach(), v.detach()) for k, v in out.past_key_values)

    return static_ids, static_mask, static_past


def stylize(Args, pages, VOICE_CACHE):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)
    terminators = [tokenizer.eos_token_id]

    static_ids, static_mask, static_past = build_system_prefix_cache(model, tokenizer)

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100):
        content = page['content']
        try:
            chunks = createChunks(content, limit=3000)

            BATCH_SIZE = min(45, len(chunks))

            prompts = generate_prompts(chunks, tokenizer)
            outputs = []
            for i in range(0, len(prompts), BATCH_SIZE):
                batch_prompts = prompts[i: i + BATCH_SIZE]
                outputs.extend(
                    paragraph_stylization(model, batch_prompts, terminators, tokenizer, static_mask, static_past)
                )

            # Save the page generated
            if outputs:
                VOICE_CACHE[page["title"]] = outputs
                updateCache('voiceCache.json', VOICE_CACHE)
                processed += 1

        except Exception as e:
            print(f"Error for page {page['title']}: {e}\n. Skipping...")

        torch.cuda.empty_cache()
    return processed


def paragraph_stylization(model, prompts, terminators, tokenizer, static_mask, static_past):
    outputs = []
    device = next(model.parameters()).device
    try:
        dyn = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        dyn_ids = dyn["input_ids"]
        dyn_mask = dyn["attention_mask"]

        batch_size = dyn_ids.size(0)
        # prefix_len = static_ids.size(1)

        # Build combined mask
        full_mask = torch.cat(
            [static_mask.repeat(batch_size, 1), dyn_mask],
            dim=1
        )

        # Broadcast prefix KV cache
        past_batch = repeat_past_kv(static_past, batch_size)

        with torch.inference_mode():
            generated = fast_generate(
                model,
                dynamic_ids=dyn_ids,
                attention_mask=full_mask,
                past_key_values=past_batch,
                max_new_tokens=512,
                eos_token_id=terminators,
            )

        gen_only = generated[:, dyn_ids.size(1):]
        for i in range(batch_size):
            text = tokenizer.decode(gen_only[i], skip_special_tokens=True).strip()
            outputs.append(text)

    except Exception as e:
        print(f"Error : {e}\n.")

    finally:
        dyn = None
        generated = None
        gen_only = None

    return outputs


def generate_prompts(chunks, tokenizer):
    prompts = []
    for paragraph in chunks:
        messages = [
            {"role": "user", "content": paragraph}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)

    return prompts

