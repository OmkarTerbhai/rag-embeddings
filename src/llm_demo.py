import os
from google import genai
from dotenv import load_dotenv;

load_dotenv();

# Initialize the client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()


# --- 2. GENERATE A TEXT RESPONSE ---
prompt = "Who is Gordon Ryan?"

text_response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)

print(f"--- Text Generation Result ---")
print(f"Answer: {text_response.text}")