#!/usr/bin/env python3
import os
import json
import requests

OPENWEBUI_BASE_URL = "https://albert-dev.beta.numerique.gouv.fr"

session = requests.Session()
session.headers.update({
    'Authorization': f'Bearer {os.environ["DEV_OPEN_WEBUI_API_KEY_SECRET"]}'
})

urlIdx = session.get(f"{OPENWEBUI_BASE_URL}/api/v1/pipelines/list").json()["data"][0]["idx"]

for pipeline_row in session.get(f"{OPENWEBUI_BASE_URL}/api/v1/pipelines/?urlIdx={urlIdx}").json()['data']:
    print(pipeline_row)
    response = session.get(f"{OPENWEBUI_BASE_URL}/api/v1/pipelines/{pipeline_row['id']}?urlIdx={urlIdx}")
    import pprint
    print(response.text)
    break
