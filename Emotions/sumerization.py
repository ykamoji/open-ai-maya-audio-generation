import re
import inspect
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Emotions.utils import fast_generate_sampling, clear_cache, getDevice
from utils import updateCache

PROMPT_PREFIX = inspect.cleandoc(f"""
You are a master storyteller blending Fantasy + Alien Sci-Fi with vivid sensory detail,
emotional resonance, and tight YA pacing. Generate **10 highly specific chapter titles**
for the user-provided chapter.

You MUST follow all rules:

CONTENT RULES
1. Titles must reflect the major events in the chapter.
2. Include powers, abilities, alien tech, or supernatural effects used.
3. Include physical actions or consequences.
4. Include emotional stakes of the characters.
5. Include motivations or internal conflicts.
6. Include a phrase that *sounds like* something a character might think or say.

STYLE RULES
- Titles must sound like they can ONLY belong to this chapter.
- No clichés, no vague mood words, no abstract nouns.
- No generic fantasy patterns.
- Titles must use concrete, sensory imagery.
- Titles must be 1-6 words.
- Do NOT use ANY of these banned words:
  shadow, dark, darkness, secret, hidden, fallen, whisper, veil,
  realm, ancient, destiny, fate, echo, storm
- If a banned word appears, remove it and rewrite the title.

EXAMPLE RESPONSES

CHAPTER 1
A boy activates an unstable alien crystal for the first time, 
it burns his palms, lights up the forest canopy, and pulls a
swarm of metallic insects toward him. He tries to shut it down
but loses control. His sister drags him away as the insects melt
through trees behind them.

TITLES:
1. Crystal Heat on My Hands
2. The Canopy Glows Too Bright
3. Flight Through Silver Wings
4. Trees Dripping Metal
5. Pulse He Can’t Shut Off

CHAPTER 2
A girl merges with a bio-engineered amphibious suit. The fusion
hurts. She dives into a flooded alien city to rescue her friend
trapped under collapsed coral structures. The water absorbs her
fear as she pushes deeper, hearing living machines shift in the walls.

TITLES:
1. Skin That Isn’t Mine
2. Descent Through Blue Ruins
3. Coral Grinding Underfoot
4. The City That Breathes Water
5. Her Fear in the Current

CHAPTER 3
A young warrior duels an intruder whose body refracts light like
glass. The intruder slices the air, bending heat, melting stone.
The warrior realizes he’s outmatched but fights to protect the
egg-shaped relic behind him. When the relic hums, both combatants
freeze.

TITLES:
1. Heat Curling Off His Blade
2. Stone Melt at My Feet
3. Fight Beside the Humming Relic
4. Glassbody Strikes Too Fast
5. When the Egg Begins to Sing

NOW CREATE TITLES FOR THE FOLLOWING CHAPTER:
""") + "\n"

PREFIX_KV_CACHE = None
PREFIX_ATTN = None


def build_prompt(content):
    return inspect.cleandoc(f"""
    {content.strip()}

    OUTPUT INSTRUCTIONS:
    Return ONLY a numbered list of **10 titles**, nothing else.

    OUTPUT:
    """)


def clean_content(text):
    text = (text
        .replace("“", '"').replace("”", '"')
        .replace("’", "'").replace("‘", "'")
        .replace("—", "-")
        .replace("…", "...")
    )

    text = re.sub(r'\n\s*\n+', '\n', text)

    text = "\n".join(line.strip() for line in text.splitlines())

    text = re.sub(r' {2,}', ' ', text)

    return text


def getSummaries(content, model, tokenizer):
    content = "\n\n".join(content)
    content = clean_content(content)

    response = set()
    try:
        enc = tokenizer(build_prompt(content), return_tensors="pt", add_special_tokens=False).to(getDevice())
        dyn_ids, dyn_attn = enc['input_ids'], enc['attention_mask']
        if dyn_attn.shape[1] == 0:
            dyn_attn = torch.ones((1, 1), device=dyn_attn.device)
            dyn_ids = torch.full((1, 1), tokenizer.eos_token_id, device=dyn_ids.device)
        full_attn = torch.cat([PREFIX_ATTN, dyn_attn], dim=1)
        with torch.inference_mode():
            generated = fast_generate_sampling(
                model,
                dynamic_ids=dyn_ids,
                attention_mask=full_attn,
                past_key_values=PREFIX_KV_CACHE,
                max_new_tokens=150,
                temperature=0.55,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.08
            )

        decoded = generated[:, dyn_ids.shape[1]:]
        output = tokenizer.decode(decoded[0], skip_special_tokens=True)
        response = extractSuggestions(output)
    except Exception as e:
        print(f"Model Error {e}.")
    finally:
        generated = None
        decoded = None
        output = None

    return response


def extractSuggestions(output):
    titles = set()
    for line in output.splitlines():
        match = re.match(r'\s*(\d+)\.\s+(.*)', line)
        if match:
            title = match.group(2).strip()
            titles.add(title)
        if len(titles) >= 10:
            break
    return titles


def summarization(model, tokenizer, pages, notebook_name, section_name, TITLE_CACHE, outputPath):

    global PREFIX_KV_CACHE, PREFIX_ATTN
    enc = tokenizer(PROMPT_PREFIX, return_tensors="pt").to(getDevice())
    prefix_ids, prefix_attn = enc['input_ids'], enc['attention_mask']
    PREFIX_ATTN = prefix_attn

    with torch.inference_mode():
        prefix_out = model(
            input_ids=prefix_ids,
            attention_mask=prefix_attn,
            use_cache=True,
        )

    PREFIX_KV_CACHE = prefix_out.past_key_values

    processed = 0
    writer = SummaryWriter(log_dir=f"{outputPath}runs/Summarization")
    for page in tqdm(pages, desc="Pages", ncols=100):
        try:
            start = time.time()
            summary = getSummaries(page['content'], model, tokenizer)
            end = time.time()
            if summary:
                title_cache = TITLE_CACHE[notebook_name][section_name].get(page["title"], {})
                content = {
                    "suggestions": set(title_cache.get("suggestions", [])),
                    "best": title_cache.get("best", "")
                }
                content["suggestions"] |= summary
                content["suggestions"] = list(content["suggestions"])
                content["suggestions"].sort()
                TITLE_CACHE[notebook_name][section_name][page["title"]] = content
                updateCache('cache/titleCache.json', TITLE_CACHE)
                processed += 1
                writer.add_scalar("Stylization/Words", sum([len(para.split()) for para in page['content']]), processed + 1)
                writer.add_scalar("Summaries/GenerationTime", (end - start), processed + 1)
            else:
                print(f"No summary found for page {page['title']}")

        except Exception as e:
            print(f"Error {e}. Skipping for {page['title']}")

        clear_cache()
    writer.flush()
    writer.close()
    return processed

