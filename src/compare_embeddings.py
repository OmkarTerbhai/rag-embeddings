import os;
from google import genai;
from dotenv import load_dotenv;
from utils.constants import text_to_embed, movies;
from scipy.spatial import distance
import numpy as np

load_dotenv();

client = genai.Client();

def create_embeddings(texts) :
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    );

    return [embedding.values for embedding in embedding_response.embeddings];

descriptions_input = [
    movie["description"] for movie in movies
];

embeddings = create_embeddings(descriptions_input);

for movie, embedding in zip(movies, embeddings) :
    movie["embedding"] = embedding;

search_text = "sea food";

search_key_embedding = create_embeddings([search_text])[0];
distances: list = [];

for movie in movies :
    dist = distance.cosine(
        search_key_embedding,
        movie["embedding"]
    );

    distances.append(dist);

min_dist_idx = np.argmin(distances);
print(movies[min_dist_idx])