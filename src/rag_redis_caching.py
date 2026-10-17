from dotenv import load_dotenv;
from langchain_core.globals import set_llm_cache;
import os;
import time;
import vertexai;
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings;
from langchain_redis import RedisSemanticCache;


load_dotenv();
os.environ.setdefault("LLM_MODEL", "gemini-3.8-flash");

if not os.environ.get("GEMINI_API_KEY") :
    raise ValueError("GEMINI KEY is not set");

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT")
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    vertexai=False
);

set_llm_cache(
    RedisSemanticCache(
        embeddings=embeddings_model,
        redis_url="redis://localhost:6379",
        distance_threshold=0.03
    )
);

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest",
                            google_api_key=os.environ.get("GEMINI_API_KEY"),
                             temperature=0);

p1 = "What is the capital of France?";
start_time = time.time();
res1 = llm.invoke(p1);

print("============First Request (API Call)============");
print(f"Response: {res1.content}");
print(f"Time taken: {time.time() - start_time:.2f} seconds");

p2 = "I am in a fun quiz, tell me what France's capital is";
start_time = time.time();
res2 = llm.invoke(p2);

print("============Second Request (Semantic Cache Hit==========");
print(f"Response: {res2.content}");
print(f"Time Taken: {time.time() - start_time:.2f} seconds");

p3 = "What is the capital of India?";
start_time = time.time();
res3 = llm.invoke(p3);

print("============Third Request (Semantic Cache Miss==========");
print(f"Response: {res3.content}");
print(f"Time Taken: {time.time() - start_time:.2f} seconds");

p4 = "What is the capital of India?";
start_time = time.time();
res4 = llm.invoke(p4);

print("============Fourth Request (Semantic Cache Hit==========");
print(f"Response: {res4.content}");
print(f"Time Taken: {time.time() - start_time:.2f} seconds");