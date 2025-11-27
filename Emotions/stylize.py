import torch
import inspect
import time
import warnings
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import repeat_past_kv, getDevice, clear_cache, slice_prefix_kv, fast_generate_sampling
from utils import updateCache
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


PREFIX_PROMPT = inspect.cleandoc("""
You are a professional editor preparing manuscripts for audiobook narration. 
Refine the paragraph ONLY for clarity, rhythm, and spoken-flow readability while keeping the author’s exact narrative style, tone and personality intact.

Editing Rules (very important):
1. Preserve all original meaning, emotional intent, atmosphere, tone, pacing, and dramatic beats.
2. Do NOT "smooth out" unique POV style. 
3. Keep POV quirks, internal monologue rhythm, sentence fragments, abrupt thoughts, slang, and informal expressions.
4. Only fix grammar, spelling, punctuation, or light sentence structure when it clearly improves clarity.
   Do NOT over-polish.
5. Do NOT replace casual, simple, vague, or informal words (e.g., "stuff", "things", "gotta", "kinda", "somewhere").
   Maintain the original diction unless fixing a typo.
6. Preserve all invented or unfamiliar nouns exactly as written.
7. Do NOT introduce new imagery, metaphors, descriptions, actions, or emotional content.
8. Use quotation marks ONLY when a character is explicitly speaking aloud.
   Internal thoughts MUST remain unquoted. Do NOT rewrite spoken lines.
9. Repetition Rule:
   - Only adjust repetition if it sounds accidentally clunky when spoken aloud.
   - Do NOT use fancy synonyms or elevated diction.
   - Do NOT remove intentional or stylistic repetition.
   - Do NOT change invented nouns or world-specific terms.

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


def stylize(model, tokenizer, pages, notebook_name, section_name, VOICE_CACHE, outputPath):

    print("Starting KV batch prefix caching.")
    static_ids, static_mask, static_past = build_system_prefix_cache(model, tokenizer)
    print("Completed KV batch prefix caching.")

    BATCH_SIZE = 20
    past_batch = repeat_past_kv(static_past, BATCH_SIZE)
    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100, position=0):
        content = page['content']
        try:
            chunks = createChunks(content, limit=2000)
            prompts = generate_prompts(chunks)
            outputs = paragraph_stylization(page["title"], model, prompts, tokenizer, static_mask, past_batch, outputPath, BATCH_SIZE)

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


def paragraph_stylization(title, model, prompts, tokenizer, static_mask, past_batch, outputPath, BATCH_SIZE):
    outputs = []
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{title}", ncols=90, position=1):
        try:
            start = time.time()
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
                generated = fast_generate_sampling(
                    model,
                    dynamic_ids=dyn_ids,
                    attention_mask=full_mask,
                    past_key_values=past_batch_val,
                    max_new_tokens=256,
                    # eos_token_id=tokenizer.eos_token_id,
                    temperature=0.15,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.05
                )
            gen_only = generated[:, dyn_ids.size(1):]
            for i in range(batch_size):
                text = tokenizer.decode(gen_only[i], skip_special_tokens=True).strip()
                outputs.append(text)
            end = time.time()
            writer.add_scalar("Stylization/Input", dyn_ids.size(1), i)
            writer.add_scalar("Stylization/GenerationTime", (end - start), i)
        except Exception as e:
            print(f"Error : {e}\n. Model didn't process the batch.")

        finally:
            dyn = None
            generated = None
            gen_only = None

    writer.flush()
    writer.close()
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

