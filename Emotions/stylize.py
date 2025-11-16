import torch
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer
from utils import updateCache

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

BATCH_SIZE = 2


def stylize(Args, pages, VOICE_CACHE):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    processed = 0
    for page in pages:
        print(f"\nRunning stylization on page {page['title']}.")
        content = page['content']

        chunks = createChunks(content, limit=5000)
        prompts = generate_prompts(chunks, tokenizer)

        outputs = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing", ncols=100):
            outputs.extend(paragraph_stlyization(i, model, prompts, terminators, tokenizer))

        torch.cuda.empty_cache()
        #Save the page generated
        if outputs:
            VOICE_CACHE[page["title"]] = outputs
            updateCache('voiceCache.json', VOICE_CACHE)
            processed += 1

    return processed


def paragraph_stlyization(i, model, prompts, terminators, tokenizer):
    outputs = []
    batch = prompts[i: i + BATCH_SIZE]

    encoded = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to('cuda')

    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )

    input_lengths = encoded["attention_mask"].sum(dim=1)

    for b in range(len(batch)):
        start = input_lengths[b]
        new_tokens = generated[b][start:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)

    return outputs


def generate_prompts(chunks, tokenizer):
    prompts = []
    for paragraph in tqdm(chunks, description="Generating prompts"):
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