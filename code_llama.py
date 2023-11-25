"""### Import"""

import transformers
import torch
import locale
import json
import os
from transformers import AutoTokenizer, TextStreamer
"""## Set up the User Interface"""
from IPython.display import display, HTML, clear_output, Markdown
from IPython.display import display, HTML
from ipywidgets import HBox, VBox
import textwrap, json
import ipywidgets as widgets
import re, time
#from google.colab import files
from pdfminer.high_level import extract_text
import io

### Select the language model
model_id = "TheBloke/Llama-2-7B-GPTQ"
# model_id = "TheBloke/Llama-2-7B-chat-GPTQ"
# model_id = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
# model_id = "TheBloke/CodeLlama-13B-Instruct-GPTQ"

# Configuration
runtimeFlag = "cuda:0" #Run on GPU (you can't run GPTQ on cpu)
cache_dir = None # by default, don't set a cache directory. This is automatically updated if you connect Google Drive.
scaling_factor = 1.0 # allows for a max sequence length of 16384*6 = 98304! Unfortunately, requires Colab Pro and a V100 or A100 to have sufficient RAM.

# Set the SYSTEM PROMPT
# DEFAULT_SYSTEM_PROMPT = 'You are a helpful pair-coding assistant.'
DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

print(SYSTEM_PROMPT)

"""### Google Drive Mounting (optional)
- Allows you to download the model to Google Drive for faster startup next time.
"""

# import os
# from google.colab import drive
# drive.mount('/content/drive')

# ## Allow the model to be saved to Google Drive for faster startup next time

# # This is the path to the Google Drive folder.
# drive_path = "/content/drive"

# # This is the path where you want to store your cache.
# cache_dir_path = os.path.join(drive_path, "My Drive/huggingface_cache")

# # Check if the Google Drive folder exists. If it does, use it as the cache_dir.
# # If not, set cache_dir to None to use the default Hugging Face cache location.
# if os.path.exists(drive_path):
#     cache_dir = cache_dir_path
#     os.makedirs(cache_dir, exist_ok=True) # Ensure the directory exists
# else:
#     cache_dir = None

# print(cache_dir)
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

"""## Load Model
If you have connected to Google Drive, the model will load from there (unless this is your first time connecting, in which case the model will be saved to Drive).
- Takes about 2 mins first time around.
- Takes about 1 min the 2nd time onwards with Google Drive.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # rope_scaling = {"type": "dynamic", "factor": scaling_factor}
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model.config.max_position_embeddings)

print(model.config)

# B_INST, E_INST = "[INST]", "[/INST]"

B_INST, E_INST = "Question: ", "Answer: "

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# max_doc_length = 50
max_context = int(model.config.max_position_embeddings*scaling_factor)
max_doc_length = int(0.75 * max_context)  # max doc length is 75% of the context length
max_doc_words = int(max_doc_length)

def generate_response(dialogs, temperature=0.01, top_p=0.9, logprobs=False):
    torch.cuda.empty_cache()
    # print(json.dumps(dialogs, indent=4))
    max_prompt_len = int(0.85 * max_context)
    max_gen_len = int(0.10 * max_prompt_len)

    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ] + dialog
        dialog_tokens = [tokenizer(
            # f"{B_INST} {B_SYS}{(dialog[0]['content']).strip()}{E_SYS}{(dialog[1]['content']).strip()} {E_INST}",
            f"{B_INST} {(dialog[1]['content']).strip()} {E_INST}", # Omits the system prompt altogether
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(runtimeFlag)]
        for i in range(2, len(dialog), 2):
            user_tokens = tokenizer(
                f"{B_INST} {(dialog[i+1]['content']).strip()} {E_INST}",
                return_tensors="pt",
                add_special_tokens=True
            ).input_ids.to(runtimeFlag)
            assistant_w_eos = dialog[i]['content'].strip() + tokenizer.eos_token
            assistant_tokens = tokenizer(
                            assistant_w_eos,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).input_ids.to(runtimeFlag)
            tokens = torch.cat([assistant_tokens, user_tokens], dim=-1)
            dialog_tokens.append(tokens)
        prompt_tokens.append(torch.cat(dialog_tokens, dim=-1))

    input_ids = prompt_tokens[0]
    if len(input_ids[0]) > max_prompt_len:
        return "\n\n **The language model's input limit has been reached. Clear the chat and start afresh!**"

    # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))

    generation_output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_new_tokens=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    );

    new_tokens = generation_output[0][input_ids.shape[-1]:]
    new_assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip();

    return new_assistant_response

def print_wrapped(text):
    # Regular expression pattern to detect code blocks
    code_pattern = r'```(.+?)```'
    matches = list(re.finditer(code_pattern, text, re.DOTALL))

    if not matches:
        # If there are no code blocks, display the entire text as Markdown
        display(Markdown(text))
        return

    start = 0
    for match in matches:
        # Display the text before the code block as Markdown
        before_code = text[start:match.start()].strip()
        if before_code:
            display(Markdown(before_code))

        # Display the code block
        code = match.group(0).strip()  # Extract code block
        display(Markdown(code))  # Display code block

        start = match.end()

    # Display the text after the last code block as Markdown
    after_code = text[start:].strip()  # Text after the last code block
    if after_code:
        display(Markdown(after_code))

dialog_history = [{"role": "system", "content": SYSTEM_PROMPT}]

button = widgets.Button(description="Send")
upload_button = widgets.Button(description="Upload .txt or .pdf")
text = widgets.Textarea(layout=widgets.Layout(width='800px'))

output_log = widgets.Output()

def on_button_clicked(b):
    user_input = text.value
    dialog_history.append({"role": "user", "content": user_input})

    text.value = ''

    # Change button description and color, and disable it
    button.description = 'Processing...'
    button.style.button_color = '#ff6e00'  # Use hex color codes for better color choices
    button.disabled = True  # Disable the button when processing

    with output_log:
        clear_output()
        for message in dialog_history:
            print_wrapped(f'**{message["role"].capitalize()}**: {message["content"]}\n')

    assistant_response = generate_response([dialog_history]);

    # Re-enable the button, reset description and color after processing
    button.description = 'Send'
    button.style.button_color = 'lightgray'
    button.disabled = False

    dialog_history.append({"role": "assistant", "content": assistant_response})

    with output_log:
        clear_output()
        for message in dialog_history:
            print_wrapped(f'**{message["role"].capitalize()}**: {message["content"]}\n')

button.on_click(on_button_clicked)

# Create an output widget for alerts
alert_out = widgets.Output()

clear_button = widgets.Button(description="Clear Chat")
text = widgets.Textarea(layout=widgets.Layout(width='800px'))

def on_clear_button_clicked(b):
    # Clear the dialog history
    dialog_history.clear()
    # Add back the initial system prompt
    dialog_history.append({"role": "system", "content": SYSTEM_PROMPT})
    # Clear the output log
    with output_log:
        clear_output()

clear_button.on_click(on_clear_button_clicked)

# Create the title with HTML
title = f"<h1 style='color: #ff6e00;'>Llama 2 Base Model 🦙</h1> <p>(Max context of: {max_context}. Uploaded files will be shortened to {max_doc_words} tokens)</p>"

# Assuming that output_log, alert_out, and text are other widgets or display elements...
first_row = HBox([button, clear_button])  # Arrange these buttons horizontally

# Arrange the two rows of buttons and other display elements vertically
layout = VBox([output_log, alert_out, text, first_row])

"""# Chat with Llama 2 or Code Llama"""

display(HTML(title))  # Use HTML function to display the title
display(layout)