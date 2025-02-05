# 5. create a simple python program to print the list of models in 
# ChatGPT, Gemini and hugging face?

import os
import getpass
from dotenv import load_dotenv # pip install python-dotenv

import google.generativeai as genai # pip install google-generativeai
import requests # pip install requests
from mistralai import Mistral # pip install mistralai


def list_gemini_models():
    # setting the Google Gemini api key
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    print("List of Google Gemini Models: ")
    gemini_model_list = genai.list_models()

    for gmodel in gemini_model_list:
        print(gmodel.name)

def list_huggingface_models():
    query = "llama"
    limit = 25
    url = f"https://huggingface.co/api/models?search={query}&limit={limit}"
    response = requests.get(url).json()
    print("\nList of Huggingface Models: ")
    for model in response:
        print("\t", model["id"])

def list_mistralai_models():
    # setting the Mistral AI api key
    if not os.environ.get("MISTRALAI_API_KEY"):
        os.environ["MISTRALAI_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")
    client = Mistral(api_key=os.environ["MISTRALAI_API_KEY"])

    list_models_response = client.models.list()
    
    print("\nList of MistralAI Models: ")
    mistral_model_dump = list_models_response.model_dump()
    for entry in mistral_model_dump['data']:
        print("\t", entry['id'])


def main():
    load_dotenv()
    #list_gemini_models()
    #list_huggingface_models()
    list_mistralai_models()

if __name__ == "__main__":
    main()