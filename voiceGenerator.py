import os
import json
import yaml
import time
from utils import CustomObject, get_yaml_loader
from GraphAPI.graphs import GraphAPI
from Generator.utils import content_stats
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert


class VoiceGenerator:

    def __init__(self):
        with open('config.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        with open('cache.json') as f:
            self.CACHE = json.load(f)

        if os.path.isfile('contentCache.json'):
            with open('contentCache.json') as f:
                self.CONTENT_CACHE = json.load(f)
        else:
            self.CONTENT_CACHE = {}
            with open('contentCache.json', 'w') as f:
                json.dump(self.CONTENT_CACHE, f)

    def updateCache(self):
        with open('contentCache.json', 'w') as f:
            json.dump(self.CONTENT_CACHE, f, indent=2, ensure_ascii=False)

    def generation(self):

        update_cache = False

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        print(f"Running voice generation for {notebook_name} {section_name}")

        pages = self.CACHE["Pages"][section_name]

        if self.Args.Generator.Pages:
            pages = pages[:self.Args.Generator.Pages]

        graph = GraphAPI(self.Args.Graph)

        for pageNo, page in enumerate(pages):
            if not self.Args.Graph.RefreshPages and self.CONTENT_CACHE and page["title"] in self.CONTENT_CACHE:
                content = self.CONTENT_CACHE[page["title"]]
            else:
                content = graph.getContent(page["id"])
                self.CONTENT_CACHE[page["title"]] = content
                update_cache = True

            if update_cache: self.updateCache()

            # print(content)

            print(f"Processing the content {page['title']} : {content_stats(content)}")

            if self.Args.Generator.OpenAI.Action:
                openAIConvert(self.Args, content, page["title"])
            elif self.Args.Generator.Maya.Action:
                mayaConvert(self.Args, content, page["title"])

            if pageNo != len(pages) - 1:
                time.sleep(30)


if __name__ == "__main__":
    VoiceGenerator().generation()