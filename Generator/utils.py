import os
import json
import torch
import shutil
from pathlib import Path
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
            print("\n" + para + "\n")

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


def create_or_load_Cache(file):
    CACHE = {}
    if os.path.isfile(file):
        with open(file) as f:
            CACHE = json.load(f)
    else:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w') as f:
            json.dump(CACHE, f)
    return CACHE


def updateCache(file, data):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_dialogues(notebook_name, section_name, page, outputPath, force_update=False, correct_voice=False):
    title = page['title']

    file = f'{outputPath}/dialogues/{notebook_name}/{section_name}/primary/{title}.json'

    DIALOGUE_CACHE = create_or_load_Cache(file)

    if not DIALOGUE_CACHE or force_update:
        chunks, tagged_list, para_breaks, broken_paras = batch_sentences(page['content'])

        DIALOGUE_CACHE = {
            "chunks": [{"part": i, "dialogue": chk, "updated": False} for i, chk in enumerate(chunks)],
            "tagged_list": tagged_list,
            "para_breaks": para_breaks,
            "broken_paras": broken_paras
        }

    edit_present = False
    if correct_voice:
        ## Load the partial cache file, if present.
        PARTIAL_CACHE = create_or_load_Cache(f'{outputPath}/dialogues/{notebook_name}/{section_name}/post/{title}.json')

        if PARTIAL_CACHE:
            edit_present = True
            ## Create a backup for confirming and restoring.
            updateCache(f'{outputPath}/backups/dialogues/{notebook_name}/{section_name}/{title}_original.json', DIALOGUE_CACHE)

            ## Edit the dialogues from the edit file
            update_voice_recreation(DIALOGUE_CACHE['chunks'], PARTIAL_CACHE["chunks"])

    updateCache(file, DIALOGUE_CACHE)

    return (
        DIALOGUE_CACHE['chunks'],
        DIALOGUE_CACHE['tagged_list'],
        DIALOGUE_CACHE['para_breaks'],
        DIALOGUE_CACHE['broken_paras'],
        edit_present
    )


def move_edited_dialogues(Graph, page, outputPath):
    notebook_name = Graph.NotebookName
    section_name = Graph.SectionName
    title = page['title']

    ## Update the updated fields of chunks to false so that run doesn't pick that again.
    primary_file = f'{outputPath}/dialogues/{notebook_name}/{section_name}/primary/{title}.json'

    primary = create_or_load_Cache(primary_file)
    for chunk in primary["chunks"]:
        chunk["updated"] = False

    updateCache(primary_file, primary)

    ## Move the edited files for confirmation
    file = f'{outputPath}/dialogues/{notebook_name}/{section_name}/post/{title}.json'
    to = f'{outputPath}/backups/dialogues/{notebook_name}/{section_name}/{title}.json'

    src = Path(file)
    dst = Path(to)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def update_voice_recreation(chunks, partial_chunks):
    for edit_chunks in partial_chunks:
        modified_chunk = {
            "part": int(edit_chunks['part']), "dialogue": edit_chunks['dialogue'], "updated": True
        }
        chunks[int(edit_chunks['part'])] = modified_chunk
