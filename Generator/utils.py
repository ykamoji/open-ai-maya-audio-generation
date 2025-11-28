import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import re

pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.)\s'


def getModels(MODEL_NAME, CACHE_PATH, platform):
    model = getARModel(MODEL_NAME, CACHE_PATH, platform)
    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)
    snac_model = getSnacModel(CACHE_PATH)
    print("Models loaded")
    return model, snac_model, tokenizer


def getARModel(MODEL_NAME, CACHE_PATH, platform, device=None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME if platform != "Kaggle" else MODEL_NAME + 'Model/',
        cache_dir=CACHE_PATH,
        torch_dtype=torch.float16,
        device_map={"": device} if device else "balanced",
        trust_remote_code=True
    )
    model.generation_config.cache_implementation = "static"
    model.config.use_cache = True
    model.eval()
    return model


def getSnacModel(CACHE_PATH):
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz",
                                      cache_dir=CACHE_PATH).eval()
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


_tag_pat = re.compile(r"\s*(<[^>]+>)\s*")


def batch_sentences(lines, limit=20):
    chunks = []
    para_breaks = []
    current_tokens = []

    def flush_current():
        if not current_tokens:
            return False
        chunk_text = " ".join(current_tokens).strip()
        chunks.append(chunk_text)
        current_tokens.clear()
        return True

    for line in lines:
        stripped = line.rstrip("\n")

        if stripped.strip() == "":
            did_flush = flush_current()
            if did_flush:
                para_breaks.append(len(chunks) - 1)
            continue

        tokens = stripped.strip()
        if not current_tokens:
            current_tokens.append(tokens)
        else:
            curr_wc = sum(len(s.split()) for s in current_tokens)
            new_wc = len(tokens.split())
            if curr_wc + new_wc > limit:
                flush_current()
                current_tokens.append(tokens)
            else:
                current_tokens.append(tokens)

    flush_current()

    if len(chunks) > 0:
        if 0 not in para_breaks:
            para_breaks.insert(0, 0)

    tagged_list = [
        bool(_tag_pat.search(chunk))
        for chunk in chunks
    ]

    return chunks, tagged_list, para_breaks


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
