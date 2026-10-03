import os;
from google import genai;
from dotenv import load_dotenv;
from utils.constants import text_to_embed;

load_dotenv();

client = genai.Client();

embedding_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text_to_embed
);

vector = embedding_response.embeddings[0].values;
print(f"____Embedding Results____");
print(f"Dimension Size: {len(vector)}");
print(f"First 5 values: {vector[:5]}");

