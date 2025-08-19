#!/usr/bin/env python3
import os
import json
import requests

OPENWEBUI_BASE_URL = "https://albert.numerique.gouv.fr"

session = requests.Session()
session.headers.update({
    'Authorization': f'Bearer {os.environ["PROD_OPEN_WEBUI_API_KEY_SECRET"]}'
})

urlIdx = session.get(f"{OPENWEBUI_BASE_URL}/api/v1/pipelines/list").json()["data"][0]["idx"]

response = session.get(f"{OPENWEBUI_BASE_URL}/api/v1/pipelines/?urlIdx={urlIdx}")
import pprint
pprint.pprint(response.json())
