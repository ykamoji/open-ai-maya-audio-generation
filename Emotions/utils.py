import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

USE_LLAMA_CPP = False

sentence_regex = re.compile(
    r'''(?x)
    (?<!\w\.\w.)          
    (?<![A-Z][a-z]\.)     
    (?<=\.|\?|!)          
    \s+                   
    '''
)


def getModelAndTokenizer(MODEL_PATH, quantize, platform):

    if USE_LLAMA_CPP:
        from llama_cpp import Llama
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            n_threads=8
        )
        return llm, None
    else:
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
        ).eval()

        model.generation_config.return_legacy_cache = True

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct" if platform != 'Kaggle' else "/kaggle/input/llama-3-1-8b-instruct/transformers/1/1/Tokenizer",
            padding_side="left",
            cache_dir=MODEL_PATH)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer


def split_sentences(text: str):
    parts = sentence_regex.split(text.strip())
    return [p.strip() for p in parts if p.strip()]