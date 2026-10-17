from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv;
from langchain_core.documents import Document
import os;
import vertexai;
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from utils.rag_docs import FACTS;

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
def to_documents(facts: list[str]) -> list[Document] :
    return [Document(page_content=text, metadata={"id": i})
            for i, text in enumerate(facts)];

D = to_documents(FACTS);
query = ("What is the cancellation policy?");
vector_db = FAISS.from_documents(D, embedding=embeddings_model);
res = vector_db.similarity_search_with_score(query, k=2);

for doc, score in res :
    print(doc.page_content,score);

FAISS_IDX_PATH = "./faiss_index";
vector_db.save_local(FAISS_IDX_PATH);


