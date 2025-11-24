import re
import os
import yaml
import json

class CustomObject:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)


def get_yaml_loader():

    pattern = re.compile(r'.*?\${(\w+)}.*?')

    def constructor_env_variables(loader, node):
        value = loader.construct_scalar(node)
        match = pattern.findall(value)
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader = yaml.SafeLoader

    loader.add_constructor('!ENV', constructor_env_variables)

    return loader


def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.split('.'))
    paras = len(content.split("\n\n"))

    return {
        "characters": count,
        "words": word_count,
        "lines": lines,
        "paragraphs": paras
    }


def create_or_load_Cache(file):
    CACHE = {}
    if os.path.isfile(file):
        with open(file) as f:
            CACHE = json.load(f)
    else:
        with open(file, 'w') as f:
            json.dump(CACHE, f)
    return CACHE


def getChapterNo(title):
    return int(re.search(r'\d+', title).group())


def create_backup(step, cache):
    updateCache(f'backups/{step}.json', cache)


def updateCache(file, data):
    with open(file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
