import os
import glob
import json
import yaml
import time
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm
from utils import create_or_load_Cache
from Emotions.stylize import stylize
from Emotions.emotions import addEmotions
from utils import CustomObject, get_yaml_loader, updateCache
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert

# Suppress env warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_THRESHOLD"] = "fatal"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class VoiceGenerator:

    def __init__(self):

        parser = argparse.ArgumentParser(description="Initidate data")
        parser.add_argument("--config", type=str, default="Default", help="Configuration file")
        parser.add_argument("--step", type=str, default="0", help="Step definition")
        args = parser.parse_args()

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.Args.Platform = args.config
        self.Args.Step = float(args.step)

        with open('contentCache.json') as f:
            self.CONTENT_CACHE = json.load(f)

        self.TITLE_CACHE = create_or_load_Cache('titleCache.json')
        self.VOICE_CACHE = create_or_load_Cache('voiceCache.json')
        self.EMOTION_CACHE = create_or_load_Cache('emotionCache.json')

    def load_content(self):
        data = []
        for pages in self.CONTENT_CACHE:
            data.append({
                "title": pages,
                "content": self.CONTENT_CACHE[pages]['content'],
            })
        return data

    def add_title(self, notebook_name, section_name, title):
        return title + " " + self.TITLE_CACHE.get(notebook_name, {}).get(section_name, {}).get(title, [None])[0]

    def generation(self):

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        pages = self.load_content()

        limit = len(pages)
        if self.Args.Generator.PageLimit:
            limit = self.Args.Generator.PageLimit

        if self.Args.Step == 1:
            print(f"\nProcessing stylization for {notebook_name} {section_name}")
            contents_to_process = []
            for pageNo, page in enumerate(pages[:limit]):
                if not self.VOICE_CACHE or page["title"] not in self.VOICE_CACHE:
                    contents_to_process.append(page)

            if contents_to_process:
                print(f"Need to stylize {len(contents_to_process)} pages")
                spell_checked_paragraphs = stylize(self.Args, contents_to_process, self.VOICE_CACHE)
                if spell_checked_paragraphs == len(contents_to_process):
                    print(f"Stylize completed!")
                else:
                    print(f"Something went wrong! Check the logs.")

        if self.Args.Step == 1.5:
            print(f"Starting post processing for voice texts.")
            for key in tqdm(self.VOICE_CACHE, desc=f"Processing content"):
                split_paragraph = False

                for idx, _ in enumerate(self.VOICE_CACHE[key]):
                    # Remove the prefix at the beginning.
                    for prefix in ["Here's the edited paragraph:\n\n", "Here's the revised paragraph:\n\n"]:
                        self.VOICE_CACHE[key][idx] = self.VOICE_CACHE[key][idx].removeprefix(prefix)

                    # Remove the extra details at the end.
                    if "\n\n" in self.VOICE_CACHE[key][idx]:
                        split_paragraph = True
                        modified_paragraphs = self.VOICE_CACHE[key][idx]
                        for para_idx, para in enumerate(self.VOICE_CACHE[key][idx].split("\n\n")):
                            if para.startswith("I made") and ("adjustments" in para or "changes" in para):
                                modified_paragraphs = "\n\n".join(modified_paragraphs.split('\n\n')[:para_idx])
                                break

                        self.VOICE_CACHE[key][idx] = modified_paragraphs

                # Keep the list paragraph seperated,
                if split_paragraph:
                    modified_paragraphs = []
                    for paras in self.VOICE_CACHE[key]:
                        parts = [part for part in paras.split("\n\n") if part.strip()]
                        modified_paragraphs.extend(parts)
                    self.VOICE_CACHE[key] = modified_paragraphs

            updateCache('voiceCache.json', self.VOICE_CACHE)
            print(f"Post processing completed voice texts.")

        if self.Args.Step == 2:
            print(f"Creating Emotions for {notebook_name} {section_name}")
            contents_to_process = []
            update_emotion_cache = False
            for pageNo, page in enumerate(pages[:limit]):
                if not self.EMOTION_CACHE or page["title"] not in self.EMOTION_CACHE:
                    contents_to_process.append(
                        {
                            "title": page["title"],
                            "content": self.VOICE_CACHE[page["title"]]
                        })
                    update_emotion_cache = True

            emotion_paragraphs = addEmotions(self.Args, contents_to_process)
            for page in emotion_paragraphs:
                suggested_title = self.add_title(notebook_name, section_name, page["title"])
                page['content'].insert(0, suggested_title)
                self.EMOTION_CACHE[page["title"]] = page['content']

            if update_emotion_cache: updateCache('emotionCache.json', self.EMOTION_CACHE)

        if self.Args.Step == 3:
            end = len(pages[:limit]) - 1
            outputPath = self.Args.Generator.AudioOutputPath.__dict__[self.Args.Platform]
            for pageNo, page in enumerate(pages[:limit]):
                print(f"Generating voice for {notebook_name} {section_name} {page['title']}")
                content = "\n\n".join(self.EMOTION_CACHE[page['title']])
                if self.Args.Generator.OpenAI.Action:
                    openAIConvert(self.Args, content, page["title"])
                elif self.Args.Generator.Maya.Action:
                    mayaConvert(self.Args, content, page["title"], outputPath)
                if pageNo != end:
                    time.sleep(60)

            audios = [file for file in glob.glob(outputPath + "audios/*.npy") if "partial" not in file]
            audios.sort(key=os.path.getmtime)
            audiobook = []
            for audio in audios:
                audiobook.append(np.load(audio))
                audiobook.append(audio)
                np.zeros(int(0.3 * 24000))
            audiobook = np.concatenate(audiobook)
            final_audio_path = outputPath + 'audios/audiobook.wav'
            sf.write(final_audio_path, audiobook, 24000)
            print(f"Saved audiobook in {final_audio_path} !")


if __name__ == "__main__":
    VoiceGenerator().generation()
