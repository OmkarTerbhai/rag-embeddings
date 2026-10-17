import google.genai as genai
import os
from dotenv import load_dotenv;

load_dotenv();

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("--- Available Models ---")
for model in client.models.list():
  if "embed" in model.name or "gemini" in model.name:
    print(model.name)