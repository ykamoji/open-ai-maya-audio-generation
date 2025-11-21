import os
import glob
import json
import random
import yaml
import re
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm

from Emotions.sumerization import summarization
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
        self.Args.Step = int(args.step)

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
        number = re.search(r"(?i)\bchat?pter\s*#?\s*(\d+)", title).group(1)
        titles = self.TITLE_CACHE.get(notebook_name, {}).get(section_name, {}).get(title, {})
        title = ""
        if "best" in titles:
            title = titles["best"]
        elif "suggestion" in titles:
            title = random.choice(titles["suggestion"])

        return f"Chapter {number}." + (f" {title}" if title else "")

    def generation(self):

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        pages = self.load_content()

        limit = len(pages)
        if self.Args.Generator.PageLimit:
            limit = self.Args.Generator.PageLimit

        if self.Args.Step >= 1:
            contents_to_process = []

            nb_cache = self.VOICE_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})

            for page in pages[:limit]:
                if page["title"] not in sec_cache:
                    contents_to_process.append(page)

            for content in contents_to_process:
                sec_cache.setdefault(content["title"], [])

            if contents_to_process:
                print(f"\nProcessing stylization for {notebook_name} {section_name}.")
                print(f"Need to stylize {len(contents_to_process)} pages")
                spell_checked_paragraphs = stylize(self.Args, contents_to_process, sec_cache)
                self.VOICE_CACHE[notebook_name][section_name] = sec_cache
                if spell_checked_paragraphs == len(contents_to_process):
                    print(f"Stylize completed!")
                else:
                    print(f"Something went wrong! Check the logs.")

        if self.Args.Step >= 2:
            print(f"Starting post processing for voice texts.")
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            post_process_paragraphs = []
            for key in tqdm(voice_cache, desc=f"Processing content"):
                split_paragraph = False
                cleaned_paragraphs = []
                for paragraph in voice_cache[key]:
                    # Remove the prefix at the beginning.
                    for prefix in ["Here's the edited paragraph:\n\n", "Here's the revised paragraph:\n\n"]:
                        paragraph = paragraph.removeprefix(prefix)

                    # Remove the extra details at the end.
                    if "\n\n" in paragraph:
                        parts = paragraph.split("\n\n")
                        for i, p in enumerate(parts):
                            if p.startswith("I made") and ("adjustments" in p or "changes" in p):
                                parts = parts[:i]
                                break
                        paragraph = "\n\n".join(parts)

                    if "\n\n" in paragraph:
                        split_paragraph = True

                    cleaned_paragraphs.append(paragraph)

                # Keep the list paragraph seperated,
                if split_paragraph:
                    final_paragraphs = []
                    for p in cleaned_paragraphs:
                        for block in p.split("\n\n"):
                            block = block.strip()
                            if block:
                                final_paragraphs.append(block)
                    cleaned_paragraphs = final_paragraphs

                post_process_paragraphs.append(cleaned_paragraphs)

            self.VOICE_CACHE[notebook_name][section_name] = post_process_paragraphs
            updateCache('voiceCache.json', self.VOICE_CACHE)
            print(f"Post processing completed voice texts.")

        if self.Args.Step >= 3 and not self.Args.Generator.SkipTitleGeneration:
            print(f"\nStarting summarization for {notebook_name} {section_name}.")
            contents_to_process = []
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            for page in pages[:limit]:
                contents_to_process.append({
                    "title": page["title"],
                    "content": voice_cache[page["title"]],
                })

            if contents_to_process:
                print(f"Need to summarize {len(contents_to_process)} pages")
                nb_cache = self.TITLE_CACHE.setdefault(notebook_name, {})
                sec_cache = nb_cache.setdefault(section_name, {})
                for content in contents_to_process:
                    if content['title'] not in sec_cache:
                        sec_cache[content['title']] = {
                            "best": "",
                            "suggestions": [],
                        }
                summarized_paragraphs = summarization(self.Args, contents_to_process, sec_cache)
                self.TITLE_CACHE[notebook_name][section_name] = sec_cache
                if summarized_paragraphs == len(contents_to_process):
                    print(f"Summarization completed!")
                else:
                    print(f"Something went wrong! Check the logs.")

        if self.Args.Step >= 4:
            print(f"Creating Emotions for {notebook_name} {section_name}")
            contents_to_process = []
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages[:limit]:
                if page["title"] not in sec_cache:
                    contents_to_process.append(
                        {
                            "title": page["title"],
                            "suggested_title": self.add_title(notebook_name, section_name, page["title"]),
                            "content": voice_cache[page["title"]]
                        })
            if contents_to_process:
                print(f"Need to add emotions to {len(contents_to_process)} pages")
                emotion_paragraphs = addEmotions(self.Args, contents_to_process, sec_cache)
                self.EMOTION_CACHE[notebook_name][section_name] = sec_cache
                if emotion_paragraphs == len(contents_to_process):
                    print(f"Emotion adding completed!")
                else:
                    print(f"Something went wrong! Check the logs.")

        if self.Args.Step == 5:
            outputPath = self.Args.Generator.AudioOutputPath.__dict__[self.Args.Platform]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages[:limit]:
                print(f"Generating voice for {notebook_name} {section_name} {page['title']}")
                content = sec_cache[page['title']]
                if self.Args.Generator.OpenAI.Action:
                    openAIConvert(self.Args, content, page["title"])
                elif self.Args.Generator.Maya.Action:
                    mayaConvert(self.Args, content, page["title"], outputPath)

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
