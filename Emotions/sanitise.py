import torch
from tqdm import tqdm
from Generator.utils import createChunks
from Emotions.utils import getModelAndTokenizer, clean_output


def generate_correct_lines(model, tokenizer, paragraph):
    # batch_prompts = "You are a grammar and spelling corrector. Answer only the corrected sentence — no explanations."\
    #         + "Example 1: this bad grammer Answer 1: This is a bad grammer." \
    #         + "Example 2: This is a good grammer. Answer 2: This is a good grammer." \
    #         + "Based on the above example and answer, correct the following sentences:" \
    #         + f"Sentences: {(",".join(lines))} Answer:"

    batch_prompts = "You are an expert English proofreader. Correct only clear spelling and grammar errors." + \
                    ("Return only the corrected paragraph. Do NOT change proper nouns, names, technical terms, "
                     "or uncommon words. Capitalize nouns and proper nouns (e.g., kuret → Kuret).") + \
                    ("Do NOT replace or guess uncommon or rare words — keep them as they are. Do NOT change word "
                     "meaning, structure, or punctuation unnecessarily. Do NOT include the instruction or the "
                     "original text.") + \
                    "Example 1: this bad grammer Answer 1: This is a bad grammar." + \
                    "Example 2: This is a good grammer. Answer 2: This is a good grammar." + \
                    "Example 3: Input: kuret was in the air. Answer 3: Kuret was in the air." \
                    "Based on the above example and answer, correct the following sentences:" + \
                    f"Paragraph: {paragraph} Answer:"

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return [d.strip() for d in decoded]


def sanitise(Args, pages):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize)

    # torch.cuda.set_per_process_memory_fraction(0.9)
    response = []
    for page in pages:
        content = page['content']
        chunks = createChunks(content, limit=5000)
        outputs = []

        for paragraph in tqdm(chunks, desc="Processing", ncols=100):
            corrected_lines = generate_correct_lines(model, tokenizer, paragraph)
            outputs.extend(corrected_lines)
            torch.cuda.empty_cache()

        corrected = clean_output(outputs, chunks)
        response.append({"title": page['title'], "content": corrected})

    return response
