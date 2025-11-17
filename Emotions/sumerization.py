import re
import json
import torch
from tqdm import tqdm
from Emotions.utils import getModelAndTokenizer
from utils import updateCache


def build_prompt(content):
    return f"""
        You are an expert YA fantasy editor.
        
        Before generating titles, extract privately:
        1. The main important events that happen
        2. Any powers, objects, or supernatural effects used
        3. Physical actions or consequences
        4. Emotional stakes for the characters
        5. Motivations 
        6. Foreshadowing hooks
        
        ---------------------------------------
        TITLE GENERATION 
        Generate **20 YA fantasy chapter titles**, grouped as:
        
        A. 5 atmospheric  
        B. 5 character-driven 
        C. 5 high-stakes 
        D. 5 foreshadowing
        
        Rules:
        1. Titles MUST be inspired by main events in the chapter YOUR INTERNAL NOTES.
        2. Titles must sound like they can ONLY belong to this chapter.
        3. No clichés or generic fantasy patterns.
        4. No vague mood words or abstract nouns or pronouns.  
        5. Absolutely avoid these words:
           shadow, dark, darkness, secret, hidden, fallen, whisper, veil,
           realm, ancient, destiny, fate, echo, storm
        6. Use sensory or concrete imagery tied to events in the chapter.
        7. 1–6 words each.
        8. Priority: imagery → emotional weight → YA rhythm.
        9. Refine the titles with better cadence, symbolism, emotional impact and uniqueness.
        
        ---------------------------------------
        FINAL OUTPUT (ONLY JSON)
        Return ONLY JSON object in this format:
        
        {{
          "Atmospheric": ["t1", "t2", "t3", "t4", "t5"],
          "CharacterDriven": ["t1", "t2", "t3", "t4", "t5"],
          "HighStakes": ["t1", "t2", "t3", "t4", "t5"],
          "Foreshadowing": ["t1", "t2", "t3", "t4", "t5"]
        }}

        ---------------------------------------
        CHAPTER:
        {content}
        ---------------------------------------
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

    json_match = re.search(r"\{[\s\S]*?\}", output, re.DOTALL)

    if not json_match:
        print("No JSON found in model output.")
        return []

    json_text = json_match.group(0)

    try:
        titles = json.loads(json_text)
    except json.JSONDecodeError:
        json_text_cleaned = json_text.replace("\n", " ").replace(",]", "]")

        try:
            titles = json.loads(json_text_cleaned)
        except:
            return []

    return titles


def summarization(Args, pages, TITLE_CACHE):

    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100):
        try:
            summary = getSummaries(page['content'], model, tokenizer)
            if summary:
                suggestions = []
                for cat in summary:
                    suggestions.extend(summary[cat])

                previous_suggestion = []
                if page['title'] in TITLE_CACHE:
                    previous_suggestion = TITLE_CACHE[page['title']]["suggestions"]
                else:
                    TITLE_CACHE[page['title']] = {}

                TITLE_CACHE[page['title']]['suggestions'] = previous_suggestion + suggestions
                updateCache('titleCache.json', TITLE_CACHE)
                processed += 1
            else:
                print(f"No summary found for page {page['title']}")

        except Exception as e:
            print(f"Error {e}. Skipping for {page['title']}")
        finally:
            torch.cuda.empty_cache()

    return processed

