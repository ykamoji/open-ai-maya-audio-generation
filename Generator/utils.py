import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import re

from Generator.Dialogues import process

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


TAG_RE = re.compile(r"<[^<>]+>")


def batch_sentences(lines, limit=25):
    paragraphs = []
    current = ""
    for line in lines[1:]:
        if line.strip() == "":
            paragraphs.append(current.strip())
            current = ""
        else:
            current += line.strip().replace("  ", " ") + " "

    if current.strip() != "":
        paragraphs.append(current.strip())

    chunks, tagged_list, para_breaks, broken_paras = process(paragraphs, limit)

    if len(broken_paras) > 0:
        print("\n\nBroken paragraphs.")
        for para in broken_paras:
            print("\n" + para +"\n")

    chunks.insert(0, lines[0])
    tagged_list.insert(0, False)
    para_breaks.insert(0, 0)

    para_breaks = para_breaks[:-1]

    br = 0
    # print("\n")
    for l, chunk in enumerate(chunks):
        # print(chunk + f" Tagged: {tagged_list[l]}")
        # print(chunk)
        num_tags = len(TAG_RE.findall(chunk))
        if num_tags > 1:
            print(f"Multiple tags in {chunk}\t\t{num_tags}")
            print("\n")

        # if br < len(para_breaks) and l == para_breaks[br]:
        #     print("------Para break-------")
        #     br += 1

    return chunks, tagged_list, para_breaks, broken_paras


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
