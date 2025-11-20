import re
import json
import torch
from tqdm import tqdm
from Emotions.utils import getModelAndTokenizer
from utils import updateCache


def build_prompt(content):
    return f"""
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
        4. Glassblood Strikes Too Fast
        5. When the Egg Begins to Sing

        NOW CREATE TITLES FOR THE FOLLOWING CHAPTER:

        {content}

        OUTPUT INSTRUCTIONS:
        Return ONLY a numbered list of **10 titles**, nothing else.

        OUTPUT:
    """.strip()


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
    prompt = build_prompt(content)
    response = []
    try:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to('cuda')

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=700,
                temperature=0.55,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = generated[:, encoded['input_ids'].shape[1]:]
        output = tokenizer.decode(decoded[0], skip_special_tokens=True)
        response = extractSuggestions(output)
    except Exception as e:
        print(f"Error {e}.")
    finally:
        del decoded, generated, encoded, output

    return response


def extractSuggestions(output):
    titles = []
    for line in output.splitlines():
        match = re.match(r'\s*(\d+)\.\s+(.*)', line)
        if match:
            title = match.group(2).strip()
            titles.append(title)
        if len(titles) >= 10:
            break
    return titles


def summarization(Args, pages, TITLE_CACHE):

    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100):
        try:
            summary = getSummaries(page['content'], model, tokenizer)
            if summary:
                previous_suggestion = []
                if page['title'] in TITLE_CACHE:
                    previous_suggestion = TITLE_CACHE[page['title']]["suggestions"]
                else:
                    TITLE_CACHE[page['title']] = {}

                TITLE_CACHE[page['title']]['suggestions'] = previous_suggestion + summary
                updateCache('titleCache.json', TITLE_CACHE)
                processed += 1
            else:
                print(f"No summary found for page {page['title']}")

        except Exception as e:
            print(f"Error {e}. Skipping for {page['title']}")
        finally:
            torch.cuda.empty_cache()

    return processed

