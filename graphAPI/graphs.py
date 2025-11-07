import os
from pydantic.utils import defaultdict
from dotenv import load_dotenv
import requests
import json
import time
import re

import graphAPI.constants as API

load_dotenv(override=True)

if "GRAPH_ACCESS_TOKEN" not in os.environ:
    raise Exception("Load Graph Access token first !")

access_token = os.getenv("GRAPH_ACCESS_TOKEN")

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}


def getNotebookList():

    response = requests.get(API.NOTEBOOK_LIST_URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Unable to get notebook list {response.status_code} {response.text}")

    json_response = json.loads(response.text)

    notebook_list = defaultdict(dict)
    if "value" in json_response:
        for notebooks in json_response["value"]:
            notebook_list[notebooks["displayName"]] = {
                "id" : notebooks["id"],
                "created" : notebooks["createdDateTime"],
                "last_modified" : notebooks["lastModifiedDateTime"],
            }

    return notebook_list

