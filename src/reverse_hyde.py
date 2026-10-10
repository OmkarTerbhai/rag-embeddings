import os
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv;

load_dotenv();
client = genai.Client();

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY")
);

vector_store = InMemoryVectorStore(embeddings_model);

document_chunks: list[str] = [
    "The Royal Awadh Kitchen specializes in Murg Parda Biryani, which is slow-cooked for exactly six hours inside a sealed clay pot wrapped in a sourdough crust.",
    "Located in the heart of Khan Market, New Delhi, this eatery features a distinct Mughlai-themed indoor courtyard decorated with 18th-century antique brass hookahs.",
    "Head Chef Arjun Kapoor infuses his signature Kashmiri Rogan Josh with a proprietary blend of 21 rare spices, including stone flower (kalpasi) sourced directly from the Western Ghats."
];

def generate_hypothetical_questions(chunk: str) -> list[str] :
    prompt = f"""
       You are preparing a vector database index for a RAG system. 
       Read the following text chunk and generate exactly 3 diverse, natural language 
       user questions that this chunk perfectly answers.
       Return ONLY the questions, one per line. Do not number or bullet them.

       TEXT CHUNK:
       {chunk}
       """;

    response = client.models.generate_content(
        model="models/gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
    ));

    questions = [ q.strip()
        for q in response.text.strip().split("\n") if q.strip()];

    return questions;

docs_to_idx = [];
for chunk in document_chunks :
    generated_questions = generate_hypothetical_questions(chunk);

    for question in generated_questions :
        doc = Document(
            page_content=question,
            metadata={
                "original_content": chunk,
                "generated_question": question
            }
        );

        docs_to_idx.append(doc);
        print(f"Embedded Question: {question}");

vector_store.add_documents(docs_to_idx);
print(f"Successfully embedded {len(docs_to_idx)} questions based on chunk");
user_query = input("Please ask....");
search_res = vector_store.similarity_search(user_query, k=1);

if search_res :
    matched_doc = search_res[0];
    print(f"User Query was: {user_query}");
    print(f"Closest matched question: {matched_doc.metadata["generated_question"]}");
    print(f"Returned Context for LLM: '{matched_doc.metadata['original_content']}'")


