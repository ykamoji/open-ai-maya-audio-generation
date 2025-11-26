import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import re

from Emotions.utils import getDevice

pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.)\s'


def getModels(MODEL_NAME, CACHE_PATH, platform):
    model = getARModel(CACHE_PATH, MODEL_NAME, platform)
    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)
    snac_model = getSnacModel(CACHE_PATH)
    print("Models loaded")
    return model, snac_model, tokenizer


def getARModel(CACHE_PATH, MODEL_NAME, platform):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME if platform != "Kaggle" else MODEL_NAME + 'Model/',
        cache_dir=CACHE_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.generation_config.cache_implementation = "static"
    model.config.use_cache = True
    model.eval()
    return model


def getSnacModel(CACHE_PATH):
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz",
                                      cache_dir=CACHE_PATH).eval()
    snac_model = snac_model.to(getDevice())
    return snac_model


def getTokenizer(MODEL_NAME, CACHE_PATH, platform):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME if platform != "Kaggle" else MODEL_NAME + 'Tokenizer/',
        cache_dir=CACHE_PATH,
        trust_remote_code=True
    )
    return tokenizer


def createChunks(content, limit=None):

    chunks = []
    paragraphs = [p.strip("\n").strip() for p in content.split("\n\n") if p.strip()]
    if limit is not None:
        chunks = paraChunks(paragraphs, limit)
    else:
        for para in paragraphs:
            lines = convert_to_sentences(para)
            chunks.extend(lines)
            chunks.append('')

    return chunks


def convert_to_sentences(content):
    return [se for se in re.split(pattern, content) if se.strip()]


def batch_sentences(lines, limit=14):
    paraBreak = "  "
    result = [lines[0]+paraBreak]
    current = ""
    lineBreak = " "
    for line in lines[1:]:
        if line.strip() == "":
            if current:
                result.append(f"{current.strip()}{paraBreak}")
                current = ""
            else:
                result[-1] += paraBreak
            continue

        if len(current.split()) + len(line.split()) > limit:
            if current:
                result.append(current.strip() + lineBreak)
            current = line
        else:
            current = (current + lineBreak + line).strip() if current else line

    if current:
        result.append(current.strip() + " ")

    return result


def paraChunks(paragraphs, limit):
    chunks = []
    for para in paragraphs:
        if len(para) >= limit:
            lines = [line for line in re.split(pattern, para) if line.strip()]
            counter = 0
            i = 0
            split_pos = [0]
            while i < len(lines):
                counter += len(lines[i])
                if counter >= limit:
                    split_pos.append(i - 1)
                    counter = len(lines[i])
                i += 1

            split_len = len(split_pos)
            for s in range(1, split_len):
                begin = split_pos[s - 1]
                end = min(split_pos[s], split_len)
                chunks.append(". ".join(lines[begin:end]))
        else:
            chunks.append(para)

    # Safe check
    for index, chunk in enumerate(chunks):
        if len(chunk) >= limit:
            msg = f"Chuck {index} {chunk[:20]} isn't under the chunk limit"
            raise Exception(msg)

    return chunks
