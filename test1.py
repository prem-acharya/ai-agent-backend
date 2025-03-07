import json

import requests

url = "http://localhost:8000/api/v1/chat"
message = "what are the available algorithm and pick one and give me code"
data = {"content": message}

headers = {"Content-type": "application/json"}

with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as r:
    for chunk in r.iter_content(1024):
        print(chunk)