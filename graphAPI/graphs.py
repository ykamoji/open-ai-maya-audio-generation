import os
from pydantic.utils import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import requests
import json
import time
import re

import graphAPI.constants as C

load_dotenv(override=True)

if "GRAPH_ACCESS_TOKEN" not in os.environ:
    raise Exception("Load Graph Access token first !")

access_token = os.getenv("GRAPH_ACCESS_TOKEN")

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}


def getNotebookList():
    print("Running Notebook List API")
    response = requests.get(C.NOTEBOOK_LIST_URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Unable to get notebook list {response.status_code} {response.text}")

    json_response = json.loads(response.text)

    notebook_list = defaultdict(dict)
    if "value" in json_response:
        for notebooks in json_response["value"]:
            notebook_list[notebooks["displayName"]] = notebooks

    return notebook_list


def getSectionList(notebook_id):
    print("Running Section List API")
    response = requests.get(C.SECTION_LIST_URL.replace("{}", notebook_id), headers=headers)
    if response.status_code != 200:
        raise Exception(f"Unable to get section list {response.status_code} {response.text}")

    json_response = json.loads(response.text)

    section_list = defaultdict(dict)
    if "value" in json_response:
        for sections in json_response["value"]:
            section_list[sections["displayName"]] = sections

    return section_list


def threadPages(section_id, skip):
    response = requests.get(C.PAGES_LIST_URL.replace("{}", section_id) + f"&skip={skip}", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Unable to get pages list {response.status_code} {response.text}")

    response = json.loads(response.text)
    return response["value"]


def getPagesList(section_id):
    print("Running Pages List API")

    response = requests.get(C.PAGES_LIST_URL.replace("{}", section_id), headers=headers)
    if response.status_code != 200:
        raise Exception(f"Unable to get pages list {response.status_code} {response.text}")

    response = json.loads(response.text)

    pages_list = []
    if "value" in response:
        pages_list = response["value"]

    total_pages = response["@odata.count"]

    pagination = range(C.PAGE_LIST_SIZE, total_pages, C.PAGE_LIST_SIZE)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(threadPages, section_id, skip) for skip in pagination]

        for f in as_completed(futures):
            pages_list.extend(f.result())

    pages_list = sorted(pages_list, key=lambda x: datetime.fromisoformat(x["createdDateTime"].replace("Z", "+00:00")))

    return pages_list
