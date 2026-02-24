import os
from ..tree import PRIMERE_ROOT
from .. import utility
import fal_client
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play, save, stream, Voice, VoiceSettings
from google import genai
from google.genai import types
import anthropic
from openai import OpenAI

def get_api_config(name: str) -> dict:
    path = os.path.join(PRIMERE_ROOT, 'json')
    fp = os.path.join(path, name)
    config_json = utility.json2tuple(fp)

    for k, v in config_json.items():
        match k:
            case "OpenAI":
                config_json['OpenAI']['APIKEY'] = os.environ.get("OPENAI_API_KEY") or config_json['OpenAI']['APIKEY']
            case "Anthropic":
                config_json['Anthropic']['APIKEY'] = os.environ.get("ANTHROPIC_API_KEY") or config_json['Anthropic']['APIKEY']
            case "BlackForest":
                config_json['BlackForest']['APIKEY'] = os.environ.get("BFL_API_KEY") or config_json['BlackForest']['APIKEY']
            case "Gemini":
                config_json['Gemini']['APIKEY'] = os.environ.get("GEMINI_API_KEY") or config_json['Gemini']['APIKEY']
            case "HeyGen":
                config_json['HeyGen']['APIKEY'] = os.environ.get("HEYGEN_API_KEY") or config_json['HeyGen']['APIKEY']
            case "FAL":
                config_json['FAL']['APIKEY'] = os.environ.get("FAL_API_KEY") or config_json['FAL']['APIKEY']
            case "Elevenlabs":
                config_json['Elevenlabs']['APIKEY'] = os.environ.get("ELEVENLABS_API_KEY") or config_json['Elevenlabs']['APIKEY']

    return config_json

def create_api_client(api_provider, config_json):
    if api_provider in config_json:
        match api_provider:
            case "OpenAI":
                APIClient = OpenAI(api_key=config_json['OpenAI']['APIKEY'])
            case "Anthropic":
                APIClient = anthropic.Anthropic(api_key=config_json['Anthropic']['APIKEY'])
            case "BlackForest":
                APIClient = True
            case "Gemini":
                APIClient = genai.Client(api_key=config_json['Gemini']['APIKEY'])
            case "HeyGen":
                APIClient = True  # genai.Client(api_key=config_json['HeyGen']['APIKEY'])
            case "FAL":
                APIClient = True
                os.environ["FAL_KEY"] = config_json['FAL']['APIKEY']
            case "Elevenlabs":
                load_dotenv()
                APIClient = ElevenLabs(api_key=config_json['Elevenlabs']['APIKEY'])
            case _:
                return (None, None)

        return (APIClient, api_provider)
    else:
        return (None, None)