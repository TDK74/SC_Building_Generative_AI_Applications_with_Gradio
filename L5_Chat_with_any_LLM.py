import os
import io
import IPython.display
import base64
import requests, json
import gradio as gr
import random

from text_generation import Client
from PIL import Image
from dotenv import load_dotenv, find_dotenv


requests.adapters.DEFAULT_TIMEOUT = 60

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']

## ------------------------------------------------------##
client = Client(os.environ['HF_API_FALCOM_BASE'],
                headers = {"Authorization" : f"Basic {hf_api_key}"}, timeout = 120)

## ------------------------------------------------------##
prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens = 256).generated_text

## ------------------------------------------------------##
def generate(input, slider):
    output = client.generate(input, max_new_tokens = slider).generated_text

    return output


demo = gr.Interface(fn = generate,
                    inputs = [gr.Textbox(label = "Prompt"),
                                gr.Slider(label = "Max new tokens",
                                          value = 20,
                                          maximum = 1024,
                                          minimum = 1)],
                    outputs = [gr.Textbox(label = "Completion")]
                   )

gr.close_all()

demo.launch(share = True, server_port = int(os.environ['PORT1']))

## ------------------------------------------------------##
def respond(message, chat_history):
        bot_message = random.choice(["Tell me more about it",
                                     "Cool, but I'm not interested",
                                     "Hmmmm, ok then"])
        chat_history.append((message, bot_message))

        return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)
    msg = gr.Textbox(label = "Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])

gr.close_all()

demo.launch(share = True, server_port = int(os.environ['PORT2']))

## ------------------------------------------------------##
def format_chat_prompt(message, chat_history):
    prompt = ""

    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"

    prompt = f"{prompt}\nUser: {message}\nAssistant:"

    return prompt


def respond(message, chat_history):
        formatted_prompt = format_chat_prompt(message, chat_history)
        bot_message = client.generate(formatted_prompt,
                                     max_new_tokens = 1024,
                                     stop_sequences = ["\nUser:",
                                                       "<|endoftext|>"]).generated_text
        chat_history.append((message, bot_message))

        return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)
    msg = gr.Textbox(label = "Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])

gr.close_all()

demo.launch(share = True, server_port = int(os.environ['PORT3']))

## ------------------------------------------------------##
def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"

    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"

    prompt = f"{prompt}\nUser: {message}\nAssistant:"

    return prompt

## ------------------------------------------------------##
def respond(message, chat_history, instruction, temperature = 0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                  max_new_tokens = 1024,
                                  stop_sequences = ["\nUser:", "<|endoftext|>"],
                                  temperature = temperature)

    acc_text = ""

    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1 : ]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]

            yield "", chat_history
            acc_text = ""

## ------------------------------------------------------##
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)
    msg = gr.Textbox(label = "Prompt")

    with gr.Accordion(label = "Advanced options", open = False):
        system = gr.Textbox(label = "System message",
                            lines = 2,
                            value = "A conversation between a user and an LLM-based AI \
                                    assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label = "temperature", minimum = 0.1,
                                maximum = 1, value = 0.7, step = 0.1)

    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])

gr.close_all()

demo.queue().launch(share = True, server_port = int(os.environ['PORT4']))

## ------------------------------------------------------##
gr.close_all()
