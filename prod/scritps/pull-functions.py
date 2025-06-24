#!/usr/bin/env python3
import os
import requests

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

response = session.get('https://albert.numerique.gouv.fr/api/v1/functions/export')

print(f'Export https://albert.numerique.gouv.fr function to "{functions_folder_path}" \n')

for function_file in response.json():
    target_path = os.path.join(
        functions_folder_path,
        f'{function_file["id"]}.py'
    )
    with open(target_path, 'w') as f:
        f.write(function_file['content'])

    print(f'- "{function_file["id"]}.py" exported')
