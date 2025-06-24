#!/usr/bin/env python3
import os
import requests

session = requests.Session()
session.headers.update({
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.environ["OPEN_WEBUI_API_KEY_SECRET"]}'
})

response = session.get('https://albert.numerique.gouv.fr/api/v1/auths/')
if response.status_code == 200:
    print(f'Hello {response.json()["name"]}, your API Key secret token works successfully')
else:
    print('Error')
    print(response.text)
