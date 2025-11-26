import os
import glob
import json
import random
import yaml
import re
import argparse
import soundfile as sf
import numpy as np

from Emotions.emotionPlacement import insertEmotions
from Emotions.postProcess import voice_post_process, emotion_det_post_process, emotion_inst_post_process
from Emotions.sumerization import summarization
from Emotions.utils import getModelAndTokenizer
from Generator.utils import getModels
from utils import create_or_load_Cache, create_backup, getChapterNo
from Emotions.stylize import stylize
from Emotions.emotionDetection import detectEmotions
from utils import CustomObject, get_yaml_loader, updateCache
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert

# Suppress env warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_THRESHOLD"] = "fatal"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setHeader(step_name):
    print("-" * 25 + step_name + "-" * 25 + "\n")


def setFooter(step_name):
    print("\n" + "-" * (50 + len(step_name)))


class VoiceGenerator:

    def __init__(self):

        parser = argparse.ArgumentParser(description="Generation")
        parser.add_argument("--config", type=str, default="Default", help="Configuration file")
        parser.add_argument("--pageLimit", type=json.loads, default=[0,-1], help="PageLimit")
        parser.add_argument("--pageNums", type=json.loads, default=None, help="List of Page Numbers to run")
        parser.add_argument("--steps", type=json.loads, default=[0], help="Step definition")
        args = parser.parse_args()

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.Args.Platform = args.config
        self.Args.Steps = args.steps

        self.Args.Generator.PageLimit = args.pageLimit
        self.PageNums = self.Args.Generator.PageNums
        self.PageNums = args.pageNums if args.pageNums else self.PageNums

        with open('cache/contentCache.json') as f:
            self.CONTENT_CACHE = json.load(f)

        self.TITLE_CACHE = create_or_load_Cache('cache/titleCache.json')
        self.VOICE_CACHE = create_or_load_Cache('cache/voiceCache.json')
        self.EMOTION_CACHE = create_or_load_Cache('cache/emotionCache.json')
        for backup in ['detection', 'detection_post', 'insertion']:
            create_or_load_Cache(f'cache/backups/{backup}.json')

        self.model, self.tokenizer = None, None
        if any(x in [1, 3, 4, 6] for x in self.Args.Steps):
            self.model, self.tokenizer = getModelAndTokenizer(self.Args)

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

        return f"Chapter {number}, " + (f" {title}" if title else "")

    def check_detection_emotion_post_process(self, title):
        lines = self.EMOTION_CACHE[self.Args.Graph.NotebookName][self.Args.Graph.SectionName][title]
        run_post_process = False
        for l in lines:
            if "output" in l:
                run_post_process = True
                break
        return run_post_process

    def check_insertion_emotion(self, title):
        lines = self.EMOTION_CACHE[self.Args.Graph.NotebookName][self.Args.Graph.SectionName][title]
        run_post_process = False
        for l in lines:
            if "messages" in l:
                run_post_process = True
                break
        return run_post_process

    def check_insertion_emotion_post_process(self, title):
        lines = self.EMOTION_CACHE[self.Args.Graph.NotebookName][self.Args.Graph.SectionName][title]
        run_post_process = False
        for l in lines:
            if type(l) == dict:
                run_post_process = True
                break
        return run_post_process

    def checkPageExist(self, step):
        cache = {}
        if step == 'voice':
            cache = self.VOICE_CACHE
        elif step == 'title':
            cache = self.TITLE_CACHE
        elif step == 'emotion':
            cache = self.EMOTION_CACHE
        section = cache.get(self.Args.Graph.NotebookName, {}).get(self.Args.Graph.SectionName, {})
        return (cache, section) if section else (None, None)

    def sort(self):
        for step in ["voice", "title", "emotion"]:
            cache, section = self.checkPageExist(step)
            if section:
                section = dict(sorted(section.items(), key=lambda page: getChapterNo(page[0])))
                cache[self.Args.Graph.NotebookName][self.Args.Graph.SectionName] = section
                updateCache("cache/" +step + "Cache.json", cache)

    def checkInPageNums(self, title):
        return self.PageNums and getChapterNo(title) in self.PageNums

    def generation(self):

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        pages = self.load_content()

        start = None
        end = None
        if self.Args.Generator.PageLimit:
            start = self.Args.Generator.PageLimit[0]
            end = self.Args.Generator.PageLimit[1]

        pages = pages[start:end]

        print(f"\nNotebook: {notebook_name}, Section: {section_name}\n")

        if 1 in self.Args.Steps:
            step_name = " Stylization "
            setHeader(step_name)
            contents_to_process = []
            nb_cache = self.VOICE_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages:
                if page["title"] not in sec_cache or self.checkInPageNums(page["title"]):
                    contents_to_process.append(page)

            for content in contents_to_process:
                sec_cache.setdefault(content["title"], [])

            if contents_to_process:
                print(f"\nProcessing {len(contents_to_process)} page(s)")
                spell_checked_paragraphs = stylize(self.model, self.tokenizer, contents_to_process, notebook_name, section_name,
                                                   self.VOICE_CACHE)
                self.sort()
                if spell_checked_paragraphs == len(contents_to_process):
                    print(f"Stylization completed!")
                else:
                    print(f"Something went wrong! Check the logs.")
            else:
                print("Nothing to process. Skipping")
            setFooter(step_name)

        if 2 in self.Args.Steps:
            step_name = " Post Processing "
            setHeader(step_name)
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            self.VOICE_CACHE[notebook_name][section_name] = voice_post_process(voice_cache)
            self.sort()
            updateCache('cache/voiceCache.json', self.VOICE_CACHE)
            print(f"Post processing voice texts completed for {notebook_name} {section_name}.")
            setFooter(step_name)

        if 3 in self.Args.Steps:
            step_name = " Title Generation "
            setHeader(step_name)
            contents_to_process = []
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            title_cache = self.TITLE_CACHE.setdefault(notebook_name, {}).setdefault(section_name, {})
            for page in pages:
                if page['title'] in voice_cache and (page["title"] not in title_cache or self.checkInPageNums(page["title"])):
                    contents_to_process.append({
                        "title": page["title"],
                        "content": voice_cache[page["title"]],
                    })

            if contents_to_process:
                print(f"Titles generation for {len(contents_to_process)} page(s)")
                summarized_paragraphs = summarization(self.model, self.tokenizer, contents_to_process, notebook_name, section_name,
                                                      self.TITLE_CACHE)
                self.sort()
                if summarized_paragraphs == len(contents_to_process):
                    print(f"Summarization completed!")
                else:
                    print(f"Something went wrong! Check the logs.")
            else:
                print("Nothing to process. Skipping")
            setFooter(step_name)

        if 4 in self.Args.Steps:
            step_name = " Emotion Detection "
            setHeader(step_name)
            contents_to_process = []
            voice_cache = self.VOICE_CACHE[notebook_name][section_name]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages:
                if page['title'] in voice_cache and (page["title"] not in sec_cache or self.checkInPageNums(page["title"])):
                    contents_to_process.append(
                        {
                            "title": page["title"],
                            "content": voice_cache[page["title"]]
                        })
            if contents_to_process:
                print(f"\nNeed to detect emotions for {len(contents_to_process)} page(s).")
                emotion_paragraphs = detectEmotions(self.model, self.tokenizer, contents_to_process, notebook_name, section_name,
                                                    self.EMOTION_CACHE)

                create_backup('detection', self.EMOTION_CACHE)
                self.sort()
                if emotion_paragraphs == len(contents_to_process):
                    print(f"Emotion detection completed!")
                else:
                    print(f"Something went wrong! Check the logs.")
            else:
                print("Nothing to process. Skipping")
            setFooter(step_name)

        if 5 in self.Args.Steps:
            step_name = " Post processing Emotions (Detection) "
            setHeader(step_name)
            emotion_cache = self.EMOTION_CACHE[notebook_name][section_name]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages:
                if self.check_detection_emotion_post_process(page["title"]) or self.checkInPageNums(page["title"]):
                    sec_cache[page["title"]] = emotion_det_post_process(emotion_cache[page["title"]], page["title"])

            create_backup('detection_post', self.EMOTION_CACHE)
            updateCache("cache/emotionCache.json", self.EMOTION_CACHE)
            self.sort()
            print(f"Post processing Emotions (Detection) completed.")
            setFooter(step_name)

        if 6 in self.Args.Steps:
            step_name = " Inserting Emotions "
            setHeader(step_name)
            contents_to_process = []
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages:
                if self.check_insertion_emotion(page["title"]) or self.checkInPageNums(page["title"]):
                    lines = []
                    for L in sec_cache[page['title']]:
                        content = {
                            'line': L['line'],
                        }
                        if 'tag' in L and L['tag']: content['tag'] = L['tag']
                        lines.append(content)

                    contents_to_process.append(
                        {
                            "title": page["title"],
                            "lines": lines
                        })
            if contents_to_process:
                print(f"\nNeed to add emotions to {len(contents_to_process)} page(s).")
                emotion_paragraphs = insertEmotions(self.model, self.tokenizer, contents_to_process, notebook_name, section_name,
                                                    self.EMOTION_CACHE)

                create_backup('insertion', self.EMOTION_CACHE)
                self.sort()
                if emotion_paragraphs == len(contents_to_process):
                    print(f"Emotion insertion completed!")
                else:
                    print(f"Something went wrong! Check the logs.")
            else:
                print("Nothing to process. Skipping")
            setFooter(step_name)

        if 7 in self.Args.Steps:
            step_name = " Post processing Emotions (Insertion) "
            setHeader(step_name)
            emotion_cache = self.EMOTION_CACHE[notebook_name][section_name]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            for page in pages:
                if self.check_insertion_emotion_post_process(page["title"]) or self.checkInPageNums(page["title"]):
                    sec_cache[page["title"]] = emotion_inst_post_process(emotion_cache[page["title"]], page["title"])
                    sec_cache[page["title"]].insert(0, self.add_title(notebook_name, section_name, page["title"]))
            updateCache("cache/emotionCache.json", self.EMOTION_CACHE)
            self.sort()
            print(f"Post processing Emotions (Insertion) completed.")
            setFooter(step_name)

        if 8 in self.Args.Steps:
            step_name = " Voice Generation "
            setHeader(step_name)

            platform = self.Args.Platform
            MayaArgs = self.Args.Generator.Maya
            MODEL_NAME = MayaArgs.ModelName.__dict__[platform]
            CACHE_PATH = MayaArgs.CachePath.__dict__[platform]
            voice_model, snac_model, voice_tokenizer = getModels(MODEL_NAME, CACHE_PATH, platform)

            outputPath = self.Args.Generator.AudioOutputPath.__dict__[platform]
            nb_cache = self.EMOTION_CACHE.setdefault(notebook_name, {})
            sec_cache = nb_cache.setdefault(section_name, {})
            audio_chapters = [file for file in glob.glob(outputPath + "audios/*.wav") if "partial" not in file]
            for page in pages:
                if page["title"] not in audio_chapters or self.checkInPageNums(page["title"]):
                    content = sec_cache[page['title']]
                    if self.Args.Generator.OpenAI.Action:
                        openAIConvert(self.Args, content, page["title"])
                    elif self.Args.Generator.Maya.Action:
                        mayaConvert(voice_model, snac_model, voice_tokenizer, MayaArgs, content, page["title"], outputPath)

            audios = [file for file in glob.glob(outputPath + "audios/*.npy") if "partial" not in file]
            audios.sort(key=os.path.getmtime)
            audiobook = []
            for audio in audios:
                audiobook.append(np.load(audio))
                audiobook.append(audio)
                np.zeros(int(0.3 * 24000))
            audiobook = np.concatenate(audiobook)
            final_audio_path = outputPath + f'audiobook.wav'
            sf.write(final_audio_path, audiobook, 24000)
            print(f"Saved audiobook in {final_audio_path} !")
            setFooter(step_name)


if __name__ == "__main__":
    VoiceGenerator().generation()
