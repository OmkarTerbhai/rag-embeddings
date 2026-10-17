from dotenv import load_dotenv;
import os;
import vertexai;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from langchain_redis import RedisVectorStore;


load_dotenv();
os.environ.setdefault("LLM_MODEL", "gemini-3.8-flash");

if not os.environ.get("GEMINI_API_KEY") :
    raise ValueError("GEMINI KEY is not set");

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT")
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY")
);

texts : list[str] = [
    "Redis is an open source in-memory data structure used as a cache and a database",
    "Google Gemini is a family of multimodal generative AI models developed by Google",
    "Langchain simplifies combining LLMs, vector stores and external tools together",
];

metadatas : list[dict[str, str]] = [
    {"source": "redis_doc"},
    {"source": "gemini_doc"},
    {"source": "langchain_doc"}
];

vector_store = RedisVectorStore.from_texts(
    texts=texts,
    embedding=embeddings_model,
    metadatas=metadatas,
    index_name="gemini-vector-index",
    redis_url="redis://localhost:6379",
    distance_metric="COSINE"
);

query = "What is redis?";
res = vector_store.similarity_search_with_score(query, k=1);

print("======Vector Search Results=====");
for doc, score in res :
    print(f" * [Score: {score:.3f}] {doc.page_content} "
          f"(Source: {doc.metadata['source']}");



