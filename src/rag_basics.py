import os;
from dotenv import load_dotenv;
from google import genai;
from langchain_core.documents import Document;
from langchain.chat_models import init_chat_model;
import vertexai;
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
from utils.rag_docs import FACTS;

load_dotenv();
os.environ.setdefault("LLM_MODEL", "gemini-3.8-flash");

if not os.environ.get("GEMINI_API_KEY") :
    raise ValueError("GEMINI KEY is not set");

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT")
)
model = init_chat_model(model=os.environ.get("LLM_MODEL"),
                        model_provider="google_vertexai");

def to_documents(facts: list[str]) -> list[Document] :
    return [Document(page_content=text, metadata={"id": i})
            for i, text in enumerate(facts)];

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY")
);

def build_store(facts: list[Document]) -> InMemoryVectorStore :
    store = InMemoryVectorStore(embedding=embeddings_model);
    store.add_documents(facts);
    return store;

vector_db = build_store(to_documents(FACTS));



def answer(query: str, store: InMemoryVectorStore , k=3) :
    relevant_facts = store.similarity_search(query=query, k=k);
    context = "\n".join(f" = {d.page_content}"
                        for d in relevant_facts);
    prompt = (
        "Answer the following question only on the notes provided\n"
        "If the notes don't contain the answer, say 'No Info found, Sorry!'\n"
        f"Notes: \n{context}\n\n Question: \n\t{query}"
    );
    client = genai.Client()
    text_response = client.models.generate_content(
        model="models/gemini-3.5-flash-lite",
        contents=prompt
    )
    return text_response.text;


QUESTION = input("Query the document corpus: ");
print("Answer with RAG: ", answer(QUESTION, vector_db));