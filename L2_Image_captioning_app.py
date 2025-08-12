import os
import io
import IPython.display
import base64
import requests, json
import gradio as gr

from PIL import Image
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']

## ------------------------------------------------------##
def get_completion(inputs, parameters = None, ENDPOINT_URL = os.environ['HF_API_ITT_BASE']):
    headers = {
              "Authorization" : f"Bearer {hf_api_key}",
              "Content-Type" : "application/json"
            }

    data = {"inputs" : inputs}

    if parameters is not None:
        data.update({"parameters" : parameters})

    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers = headers,
                                data = json.dumps(data))

    return json.loads(response.content.decode("utf-8"))

## ------------------------------------------------------##
image_url = "https://free-images.com/sm/9596/dog_animal_greyhound_983023.jpg"
display(IPython.display.Image(url = image_url))
get_completion(image_url)

## ------------------------------------------------------##
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format = 'PNG')
    byte_arr = byte_arr.getvalue()

    return str(base64.b64encode(byte_arr).decode('utf-8'))


def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)

    return result[0]['generated_text']


gr.close_all()

demo = gr.Interface(fn = captioner,
                    inputs = [gr.Image(label = "Upload image", type = "pil")],
                    outputs = [gr.Textbox(label = "Caption")],
                    title = "Image Captioning with BLIP",
                    description = "Caption any image using the BLIP model",
                    allow_flagging = "never",
                    examples = ["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

demo.launch(share = True, server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
gr.close_all()
