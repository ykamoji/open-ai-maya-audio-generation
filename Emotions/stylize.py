import torch
import inspect
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer, fast_generate, repeat_past_kv, getDevice, clear_cache, slice_prefix_kv
from utils import updateCache
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


PREFIX_PROMPT = inspect.cleandoc("""
You are a professional editor preparing manuscripts for audiobook narration. Refine the paragraph to improve clarity, rhythm, 
and spoken-flow readability while maintaining all story details and staying true to the author’s voice.

Editing Rules (very important):
- Preserve the emotional intent, atmosphere, tone, and the author's voice.
- Maintain pacing and dramatic beats; do not remove key details or shorten the scene.
- Fix grammar, sentence structure, punctuation, and word choice **only** when it clearly improves clarity or how the text sounds when read aloud.
- Keep vague or informal language as-is (e.g., "stuff","things","somewhere"); do not make the prose generic or over-polished.
- Do not introduce new imagery, metaphors, actions, or descriptive elements.
- Only apply quotation marks when the text explicitly or implicitly indicates that a character is speaking aloud, and preserve the original wording of the spoken line.
- Keep the writing consistent across sentences and scenes, improving readability without changing meaning.

Return only the edited paragraph. If no edits are needed, return it unchanged.
""") + "\n"


def build_system_prefix_cache(model, tokenizer):

    enc = tokenizer(PREFIX_PROMPT, return_tensors="pt").to(getDevice())

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


def stylize(model, tokenizer, pages, notebook_name, section_name, VOICE_CACHE):

    static_ids, static_mask, static_past = build_system_prefix_cache(model, tokenizer)

    BATCH_SIZE = 20
    past_batch = repeat_past_kv(static_past, BATCH_SIZE)
    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100, position=0):
        content = page['content']
        try:
            chunks = createChunks(content, limit=2000)
            prompts = generate_prompts(chunks)
            outputs = paragraph_stylization(page["title"], model, prompts, tokenizer, static_mask, past_batch, BATCH_SIZE)

            # Save the page generated
            if outputs:
                VOICE_CACHE[notebook_name][section_name][page["title"]] = outputs
                updateCache('cache/voiceCache.json', VOICE_CACHE)
                processed += 1
            else:
                print(f"Stylization skipped for page {page['title']}.")

        except Exception as e:
            print(f"Error for page {page['title']}: {e}\n. Skipping...")

        clear_cache()
    return processed


def paragraph_stylization(title, model, prompts, tokenizer, static_mask, past_batch, BATCH_SIZE):
    outputs = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{title}", ncols=90, position=1):
        try:
            batch_prompts = prompts[i: i + BATCH_SIZE]
            dyn = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
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

            past_batch_val = past_batch
            if batch_size < BATCH_SIZE:
                past_batch_val = slice_prefix_kv(past_batch, batch_size)

            with torch.inference_mode():
                generated = fast_generate(
                    model,
                    dynamic_ids=dyn_ids,
                    attention_mask=full_mask,
                    past_key_values=past_batch_val,
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_only = generated[:, dyn_ids.size(1):]
            for i in range(batch_size):
                text = tokenizer.decode(gen_only[i], skip_special_tokens=True).strip()
                outputs.append(text)

        except Exception as e:
            print(f"Error : {e}\n. Model didn't process the batch.")

        finally:
            dyn = None
            generated = None
            gen_only = None

    return outputs


def generate_prompts(chunks):
    prompts = []
    for paragraph in chunks:
        prompt = inspect.cleandoc(f"""
                Paragraph:
                {paragraph}

                Edited paragraph:
                """)
        prompts.append(prompt)
    return prompts

