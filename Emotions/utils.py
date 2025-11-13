import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

ALLOWED_TAGS = [
    "laugh","laugh_harder","sigh","chuckle","gasp","angry","excited","whisper",
    "cry","scream","sing","snort","exhale","gulp","giggle","sarcastic","curious"
]

TAG_PENALTY_WEIGHTS = {
    "sigh": 1.0,
    "chuckle": 0.8,
    "giggle": 0.7,
    "gasp": 1.0,
    "angry": 1.6,
    "sarcastic": 1.8,
    "cry": 1.3,
    "scream": 2.0,
    "sing": 2.0,
    "snort": 1.4,
    "laugh": 1.0,
    "laugh_harder": 1.2,
    "exhale": 1.0,
    "gulp": 1.1,
    "whisper": 1.0,
    "excited": 1.1,
    "curious": 1.0,
}

TAG_SIMILARITY_WEIGHTS = {
    "laugh":         ["laugh_harder", "giggle", "chuckle"],
    "laugh_harder":  ["laugh", "giggle"],
    "giggle":        ["laugh", "laugh_harder", "chuckle"],
    "chuckle":       ["laugh", "giggle"],

    "cry":           ["sigh", "exhale"],
    "sigh":          ["cry", "exhale"],
    "exhale":        ["sigh", "cry"],

    "angry":         ["sarcastic", "scream", "snort"],
    "sarcastic":     ["angry", "snort"],
    "snort":         ["sarcastic", "laugh"],

    "gasp":          ["excited", "cry"],
    "excited":       ["gasp", "laugh"],

    "whisper":       ["curious"],
    "curious":       ["whisper", "excited"],

    "gulp":          ["exhale"],
    "scream":        ["angry"],
    "sing":          ["excited", "laugh"],
}


AUTO_CORRECT_MAP = {
    "happy": "giggle",
    "joy": "giggle",
    "joyful": "giggle",
    "sad": "cry",
    "upset": "cry",
    "frustrated": "sigh",
    "annoyed": "sigh",
    "mad": "angry",
    "furious": "angry",
    "angry_sigh": "angry",
    "rage": "angry",
    "surprised": "gasp",
    "shocked": "gasp",
    "wow": "gasp",
    "soft": "whisper",
    "quiet": "whisper",
    "unsure": "curious",
    "confused": "curious",
    "thinking": "curious",
    "singing": "sing",
    "laughing": "laugh",
    "lol": "laugh",
    "haha": "laugh",
}


def getModelAndTokenizer(MODEL_PATH, quantize):

    bnb_config = None
    if quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        # dtype="float16",
        low_cpu_mem_usage=True,
        cache_dir=MODEL_PATH
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                              cache_dir=MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def clean_output(outputs, chunks):

    extracted_data = []
    for idx, output in enumerate(outputs):
        if "Answer:" in output:
            clean_lines = output.rsplit("Answer:", 1)[-1].strip()
            extracted_data.append(clean_lines)
        else:
            print(f"\nFix needed for {output} !\n")
            # backup in case the Model fails
            extracted_data.append(chunks[idx])

    return extracted_data


def convert_to_sentences(content):
    lines = []
    for para in content:
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.)\s'
        for se in re.split(pattern, para):
            lines.append(re.sub(r'(^|\s)\d+\.\s*', r'\1', se))
    return lines

