import os
import glob
import json
import yaml
import time
import argparse
import soundfile as sf
import numpy as np
from utils import createCache
from Emotions.sanitise import sanitise
from Emotions.emotions import addEmotions
from utils import CustomObject, get_yaml_loader, updateCache
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert


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

        self.TITLE_CACHE = createCache('titleCache.json')
        self.VOICE_CACHE = createCache('voiceCache.json')
        self.EMOTION_CACHE = createCache('emotionCache.json')

    def load_content(self):
        data = []
        for pages in self.CONTENT_CACHE:
            data.append({
                "title": pages,
                "content": self.CONTENT_CACHE[pages]['content'],
            })
        return data

    def generation(self):

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        pages = self.load_content()

        limit = len(pages)
        if self.Args.Generator.PageLimit:
            limit = self.Args.Generator.PageLimit

        if self.Args.Step == 1:
            print(f"Processing spellcheck and grammars for {notebook_name} {section_name}")
            contents_to_process = []
            update_voice_cache = False
            for pageNo, page in enumerate(pages[:limit]):
                if not self.VOICE_CACHE or page["title"] not in self.VOICE_CACHE:
                    contents_to_process.append(page)
                    update_voice_cache = True

            print(f"Need to run spellcheck and grammars for {len(contents_to_process)} pages")
            spell_checked_paragraphs = sanitise(self.Args, contents_to_process)
            for page in spell_checked_paragraphs:
                page['content'].insert(1, self.TITLE_CACHE[notebook_name][section_name][page["title"]][0])
                self.VOICE_CACHE[page["title"]] = page['content']

            if update_voice_cache: updateCache('voiceCache.json', self.VOICE_CACHE)

            print(f"Spell check and grammar completed !")

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
                self.EMOTION_CACHE[page["title"]] = page['content']

            if update_emotion_cache: updateCache('emotionCache.json', self.EMOTION_CACHE)

        if self.Args.Step == 3:
            end = len(self.EMOTION_CACHE[:limit]) - 1
            outputPath = self.Args.Generator.AudioOutputPath.__dict__[self.Args.Platform]
            for pageNo, page in enumerate(self.EMOTION_CACHE[:limit]):
                print(f"Generating voice for {notebook_name} {section_name} {page['title']}")
                if self.Args.Generator.OpenAI.Action:
                    openAIConvert(self.Args, page['content'], page["title"])
                elif self.Args.Generator.Maya.Action:
                    mayaConvert(self.Args, page['content'], page["title"], outputPath)
                if pageNo != end:
                    time.sleep(60)

            audios = glob.glob(outputPath + "audios/*.npy")
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