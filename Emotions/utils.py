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
    "surprise": "gasp",
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
    "concerned": "curious",
    "disappointed": "sigh",
    "sadness": "cry",
    "anxious": "gulp",
    "nervous": "gulp",
    "tense": "gulp",
    "relieved": "exhale",
    "suspicious": "curious",
    "thrilled": "excited"
}


def getModelAndTokenizer(MODEL_PATH, quantize, platform):

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16

    bnb_config = None
    if quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct" if platform != 'Kaggle' else "/kaggle/input/llama-3-1-8b-instruct/transformers/1/1/Model",
        cache_dir=MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=DTYPE,
        load_in_4bit=False,
        load_in_8bit=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct" if platform != 'Kaggle' else "/kaggle/input/llama-3-1-8b-instruct/transformers/1/1/Tokenizer",
        padding_side="left",
        cache_dir=MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

