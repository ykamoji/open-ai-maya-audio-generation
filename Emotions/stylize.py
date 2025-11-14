import torch
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer


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


def generate_paragrpah(model, tokenizer, paragraph):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": paragraph}
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = output_ids[0][input_ids.shape[-1]:]

    return tokenizer.decode(decoded, skip_special_tokens=True).strip()


def stylize(Args, pages):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    response = []
    for page in pages:
        content = page['content']
        chunks = createChunks(content, limit=5000)
        outputs = []
        for paragraph in tqdm(chunks, desc="Processing", ncols=100):
            stylized_paragraph = generate_paragrpah(model, tokenizer, paragraph)
            outputs.append(stylized_paragraph)
        torch.cuda.empty_cache()
        response.append({"title": page['title'], "content": outputs})

    return response
