import torch
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer
from utils import updateCache
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


SYSTEM_PROMPT = """
You are a professional editor preparing manuscripts for audiobook narration. Your task is to refine the paragraph to improve clarity, rhythm, 
and spoken-flow readability while maintaining all story details and staying true to the author’s voice.

Very important Guidelines:
- Preserve the emotional intent, atmosphere, and style of the writing.
- Maintain the pacing and dramatic beats of the scene.
- Keep character voice, tone, and attitude unchanged.
- Correct grammar, sentence structure, punctuation, and word choice only where it 
  enhances clarity or improves how the text sounds when read aloud.
- Strengthen readability and natural rhythm for voice actors without altering meaning.
- Avoid introducing new imagery, metaphors, or descriptive elements.
- Avoid removing key details or shortening passages in ways that alter the feel.
- Keep the writing consistent from sentence to sentence and across scenes.
- Do not over-polish or make the prose sound generic or mechanical.

Return only the edited paragraph, clean and ready for audiobook production.
""".strip()


def stylize(Args, pages, VOICE_CACHE):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=100):
        content = page['content']
        try:
            chunks = createChunks(content, limit=5000)
            prompts = generate_prompts(chunks, tokenizer)
            outputs = paragraph_stylization(model, prompts, terminators, tokenizer)

            # Save the page generated
            if outputs:
                VOICE_CACHE[page["title"]] = outputs
                updateCache('voiceCache.json', VOICE_CACHE)
                processed += 1

        except Exception as e:
            print(f"Error for page {page['title']}: {e}\n. Skipping...")
        finally:
            torch.cuda.empty_cache()

    return processed


def paragraph_stylization(model, prompts, terminators, tokenizer):
    outputs = []
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to('cuda')

        with torch.inference_mode():
            generated = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=True
            )

        prompt_len = encoded["input_ids"].shape[1]
        sequences = generated.sequences
        generated_content = sequences[:, prompt_len:]

        for b in range(len(prompts)):
            text = tokenizer.decode(generated_content[b], skip_special_tokens=True).strip()
            outputs.append(text)

    except Exception as e:
        print(f"Error : {e}\n.")

    finally:
        del encoded, generated, sequences, generated_content

    return outputs


def generate_prompts(chunks, tokenizer):
    prompts = []
    for paragraph in chunks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paragraph}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        prompts.append(prompt_text)
    return prompts