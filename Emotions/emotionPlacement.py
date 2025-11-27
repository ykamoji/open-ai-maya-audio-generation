import torch
import inspect
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Emotions.utils import clear_cache, fast_generate, slice_prefix_kv, repeat_past_kv
from utils import updateCache


PLACEMENT_PREFIX = inspect.cleandoc(f"""
        You are an expert at placing TTS emotion tags into sentences.

        Your job is to choose the BEST position to insert the tag inside the TARGET sentence, 
        without modifying any words. The tag will be inserted BEFORE the chosen word index.

        RULES:
        1. If the emotional cue is direct or indirect
           (examples: wide-eyed, shocked, tense, uneasy, startled, frozen, nervous, hesitant, trembling),
        3. Place the tag at the strongest emotional point of the sentence.
           - If multiple cues exist, choose the most intense one.
           - If intensity is unclear, choose the earliest cue.
        2. You may position the tag either BEFORE or AFTER the emotional cue word. 
           - Default to BEFORE unless AFTER clearly improves delivery (e.g., after a comma or natural pause).
        3. If the emotional moment resolves at the end of the sentence, return {{ "position": "END" }}.
        4. If unsure or the sentence has no emotional cue → {{ "position": "" }}.
        5. Words are indexed using whitespace tokenization.
           Punctuation attached to a word (comma, period, quotes) counts as part of that word.
        6. Do NOT rewrite the sentence.
        7. Do NOT output explanations or anything else.
        
        OUTPUT FORMAT (STRICT):
            - If a single integer index (0-based) → {{ "position": "index" }}
            - If END → {{ "position": "END" }}
            - If unsure or the sentence has no emotional cue → {{ "position": "" }}

        EXAMPLE 1:
        TAG: EXCITED
        TARGET SENTENCE:
        A spark of anticipation flickered through her as she stepped toward the doorway.
        OUTPUT: {{"position": 5}}

        EXAMPLE 2:
        TAG: EXHALE
        TARGET SENTENCE:
        He paused for a moment, a nervous tremor in his voice before he continued speaking.
        OUTPUT: {{"position": 11}}

        EXAMPLE 3:
        TAG: SIGH
        TARGET SENTENCE:
        She lowered her shoulders as the weight of the moment settled heavily on her.
        OUTPUT: {{"position": "END"}}
        """) + "\n"


def build_placement_dynamic(tag, sentence):
    return inspect.cleandoc(f"""
        TAG: {tag}
        TARGET SENTENCE:
        {sentence}
        OUTPUT:
    """)


PLACEMENT_STATIC_MASK = None
PLACEMENT_STATIC_BATCH_PAST = None


def init_static_placement_cache(tokenizer, model, BATCH_SIZE):

    global PLACEMENT_STATIC_MASK, PLACEMENT_STATIC_BATCH_PAST

    if PLACEMENT_STATIC_MASK is not None and PLACEMENT_STATIC_BATCH_PAST is not None:
        return  # already initialized

    device = next(model.parameters()).device

    enc = tokenizer(PLACEMENT_PREFIX, return_tensors="pt").to(device)
    static_ids = enc["input_ids"]  # (1, prefix_len)
    PLACEMENT_STATIC_MASK = enc["attention_mask"]  # (1, prefix_len)

    with torch.inference_mode():
        out = model(
            input_ids=static_ids,
            attention_mask=PLACEMENT_STATIC_MASK,
            use_cache=True,
        )

    # detach so we don't track autograd history
    placement_static_past = tuple((k.detach(), v.detach()) for k, v in out.past_key_values)

    PLACEMENT_STATIC_BATCH_PAST = repeat_past_kv(placement_static_past, BATCH_SIZE)


def insert_emotion_index(title, sentences, tags, model, tokenizer, outputPath, BATCH_SIZE):
    N = len(sentences)
    device = next(model.parameters()).device
    placement_indexes = []

    global PLACEMENT_STATIC_MASK, PLACEMENT_STATIC_BATCH_PAST

    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    for i in tqdm(range(0, N, BATCH_SIZE), desc=f"{title}", ncols=90, position=1):
        batch_sentences = sentences[i:i + BATCH_SIZE]
        batch_tags = tags[i:i + BATCH_SIZE]
        start_time = time.time()
        dynamic_texts = [build_placement_dynamic(t, s) for t, s in zip(batch_tags, batch_sentences)]

        dyn_enc = tokenizer(dynamic_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        dyn_ids = dyn_enc["input_ids"]  # (N, D)
        dyn_mask = dyn_enc["attention_mask"]  # (N, D)

        batch_size = dyn_ids.size(0)

        static_mask = PLACEMENT_STATIC_MASK.repeat(batch_size, 1)  # (B, prefix_len)
        full_mask = torch.cat([static_mask, dyn_mask], dim=1)

        past_batch_val = PLACEMENT_STATIC_BATCH_PAST
        if batch_size < BATCH_SIZE:
            past_batch_val = slice_prefix_kv(PLACEMENT_STATIC_BATCH_PAST, batch_size)

        try:
            with torch.inference_mode():
                out_ids = fast_generate(
                    model,
                    dynamic_ids=dyn_ids,
                    attention_mask=full_mask,
                    past_key_values=past_batch_val,
                    max_new_tokens=10,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Strip dynamic prompt only (prefix is not in out_ids)
            gen_only = out_ids[:, dyn_ids.shape[1]:]

            decoded_batch = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for i in range(batch_size):
                placement_indexes.append(decoded_batch[i])

        except Exception as e:
            print(f"Exception: {e}. Defaulting emotion to the end of sentence in this batch")
            for i in range(batch_size):
                placement_indexes.append("END")

        finally:
            decoded_batch = None
            out_ids = None
            gen_only = None
            full_mask = None
            past_key_values = None
            batch_past = None
            clear_cache()
        end = time.time()
        writer.add_scalar("EmotionPlacement/GenerationTime", (end - start_time), i+1)

    writer.flush()
    writer.close()

    return placement_indexes


def insertEmotions(model, tokenizer, pages, notebook_name, section_name, EMOTION_CACHE, outputPath):

    BATCH_SIZE = 20
    print("Starting KV batch prefix caching.")
    init_static_placement_cache(tokenizer, model, BATCH_SIZE)
    print("Completed KV batch prefix caching.")
    progress = 0
    for page in tqdm(pages, desc="Pages", ncols=100, position=0):
        try:
            insert_line_pos = []
            sentences = []
            tags = []
            lines = []
            for pos, L in enumerate(page['lines']):
                if 'tag' in L and L['tag']:
                    sentences.append(L['line'])
                    tags.append(L['tag'])
                    insert_line_pos.append(pos)
                else:
                    lines.append(L['line'])

            if sentences:
                placement_indexes = insert_emotion_index(page['title'], sentences, tags, model, tokenizer, outputPath, BATCH_SIZE)
                if placement_indexes:
                    for idx, placement in enumerate(placement_indexes):
                        lines.insert(insert_line_pos[idx], {
                            'line': sentences[idx],
                            'tag': tags[idx],
                            'pos': placement
                        })
                else:
                    print(f"No placements for the emotions. Skipping {page['lines']}")

            EMOTION_CACHE[notebook_name][section_name][page['title']] = lines
            updateCache('cache/emotionCache.json', EMOTION_CACHE)
            progress += 1
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}.")

    global PLACEMENT_STATIC_MASK, PLACEMENT_STATIC_BATCH_PAST
    PLACEMENT_STATIC_MASK = None
    PLACEMENT_STATIC_BATCH_PAST = None
    clear_cache()

    return progress
