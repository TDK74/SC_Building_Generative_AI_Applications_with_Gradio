import os
import io
import base64
import requests, json
import gradio as gr

from IPython.display import Image, display, HTML
from PIL import Image
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']

## ------------------------------------------------------##
def get_completion(inputs, parameters = None, ENDPOINT_URL = os.environ['HF_API_SUMMARY_BASE']):
    headers = {
              "Authorization" : f"Bearer {hf_api_key}",
              "Content-Type" : "application/json"
            }

    data = { "inputs": inputs }

    if parameters is not None:
        data.update({"parameters" : parameters})

    response = requests.request("POST",
                                ENDPOINT_URL, headers = headers,
                                data = json.dumps(data)
                               )

    return json.loads(response.content.decode("utf-8"))

## ------------------------------------------------------##
text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
        as an 81-storey building, and the tallest structure in Paris.
        Its base is square, measuring 125 metres (410 ft) on each side.
        During its construction, the Eiffel Tower surpassed the Washington
        Monument to become the tallest man-made structure in the world,
        a title it held for 41 years until the Chrysler Building
        in New York City was finished in 1930. It was the first structure
        to reach a height of 300 metres. Due to the addition of a broadcasting
        aerial at the top of the tower in 1957, it is now taller than the
        Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the
        Eiffel Tower is the second tallest free-standing structure in France
        after the Millau Viaduct.''')

get_completion(text)

## ------------------------------------------------------##
def summarize(input):
    output = get_completion(input)

    return output[0]['summary_text']


gr.close_all()
demo = gr.Interface(fn = summarize, inputs = "text", outputs = "text")
demo.launch(share = True, server_port = int(os.environ['PORT1']))
