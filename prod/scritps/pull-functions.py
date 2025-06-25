#!/usr/bin/env python3
import os
import json
import requests

OPENWEBUI_BASE_URL = "https://albert.numerique.gouv.fr"

session = requests.Session()
session.headers.update({
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.environ["PROD_OPEN_WEBUI_API_KEY_SECRET"]}'
})

functions_folder_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'functions'
    )
)

response = session.get(f'{OPENWEBUI_BASE_URL}/api/v1/functions/export')

print(f'Export {OPENWEBUI_BASE_URL} function to "{functions_folder_path}" \n')

for function_row in response.json():
    with open(
        os.path.join(
            functions_folder_path,
            f'{function_row["id"]}.py'
        ),
        'w'
    ) as f:
        f.write(function_row['content'])

    with open(
        os.path.join(
            functions_folder_path,
            'valves',
            f'{function_row["id"]}.json.secret'
        ),
        'w'
    ) as f:
        valve_export_response = session.get(f'{OPENWEBUI_BASE_URL}/api/v1/functions/id/{function_row["id"]}/valves')
        if valve_export_response.status_code == 200:
            f.write(
                json.dumps(
                    valve_export_response.json(),
                    indent=4
                )
            )
        else:
            print(f'Error: {valve_export_response.text}')
            break

    print(f'- "{function_row["id"]}" function exported')
