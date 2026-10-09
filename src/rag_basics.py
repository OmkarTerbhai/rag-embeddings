import os;
from dotenv import load_dotenv;
from google import genai;
from langchain_core.documents import Document;
from langchain.chat_models import init_chat_model;
import vertexai;
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings;

load_dotenv();
os.environ.setdefault("LLM_MODEL", "gemini-3.8-flash");

if not os.environ.get("GEMINI_API_KEY") :
    raise ValueError("GEMINI KEY is not set");


FACTS: list[str] = [
    "Jon Jones is widely considered one of the greatest mixed martial artists of all time, holding the record for the youngest champion in UFC history at age 23.",
    "The Ultimate Fighting Championship (UFC) was founded in 1993 by Art Davie, Rorion Gracie, Bob Meyrowitz, Campbell McLaren, and David Isaacs.",
    "UFC 1 took place on November 12, 1993, in Denver, Colorado, and featured an eight-man tournament format with no weight classes or time limits.",
    "Royce Gracie won the tournament at UFC 1, UFC 2, and UFC 4, showcasing the effectiveness of Brazilian Jiu-Jitsu (BJJ) against larger opponents.",
    "Khabib Nurmagomedov retired as the undefeated UFC Lightweight Champion with a professional record of 29-0 after defeating Justin Gaethje at UFC 254.",
    "Conor McGregor became the first simultaneous two-division champion in UFC history by defeating Eddie Alvarez for the lightweight title at UFC 205 while holding the featherweight belt.",
    "Amanda Nunes is regarded as the greatest female fighter in MMA history, having held both the UFC Women's Bantamweight and Featherweight championships simultaneously.",
    "The UFC Octagon is a signature eight-sided fenced enclosure with walls made of chain-link fence coated in black vinyl, measuring 30 feet across.",
    "UFC weight classes for men include Flyweight (125 lbs), Bantamweight (135 lbs), Featherweight (145 lbs), Lightweight (155 lbs), Welterweight (170 lbs), Middleweight (185 lbs), Light Heavyweight (205 lbs), and Heavyweight (265 lbs).",
    "Chael Sonnen famously challenged Anderson Silva for the UFC Middleweight Championship at UFC 117, dominating four rounds before being submitted via triangle armbar in the fifth.",
    "Demetrious Johnson holds the record for the most consecutive title defenses in UFC history with 11 successful defenses of his Flyweight championship.",
    "The BMF (Baddest Motherf**ker) championship belt was introduced at UFC 244, where Jorge Masvidal defeated Nate Diaz via doctor stoppage.",
    "Max Holloway won the symbolic BMF title at UFC 300 by knocking out Justin Gaethje with a spectacular last-second exchange in the fifth round.",
    "Kamaru Usman defended his Welterweight title five times before being knocked out by Leon Edwards' famous head kick at UFC 278 in Salt Lake City.",
    "Israel Adesanya reclaimed his Middleweight title at UFC 287 by knocking out his long-time kickboxing and MMA rival Alex Pereira in the second round.",
    "Alex Pereira achieved the rare feat of becoming a two-division UFC champion in record time, capturing the Middleweight belt and later the Light Heavyweight belt.",
    "The UFC Performance Institute (UFC PI), located in Las Vegas, Nevada, provides state-of-the-art training, nutrition, and physical therapy facilities for rostered athletes.",
    "Dana White has served as the President and later Chief Executive Officer of the UFC, instrumental in growing it from a struggling promotion into a multi-billion dollar global brand.",
    "Georges St-Pierre avenged both of his career losses by defeating Matt Hughes and Matt Serra, cementing his status as a legendary dominant UFC Welterweight Champion.",
    "UFC championship bouts, along with modern pay-per-view main events, are scheduled for five rounds of five minutes each, while standard bouts are three rounds.",
    "Stipe Miocic holds the record for the most consecutive heavyweight title defenses in UFC history, defending the belt three times during his first reign.",
    "Islam Makhachev captured the vacant UFC Lightweight Championship at UFC 280 by submitting Charles Oliveira, continuing the dominance of the Dagestani wrestling style.",
    "The UFC was acquired by Endeavor (formerly WME-IMG) in 2016 for approximately $4 billion, marking one of the largest transactions in sports history.",
    "Sean O'Malley won the UFC Bantamweight Championship by delivering a precise counter right hand to knock out Aljamain Sterling at UFC 292.",
    "The Unified Rules of Mixed Martial Arts govern modern UFC bouts, outlining illegal strikes such as 12-to-6 elbows, groin strikes, and strikes to the spine or back of the head.",
    "Ilia Topuria became the UFC Featherweight Champion by knocking out the long-reigning champion Alexander Volkanovski in the second round at UFC 298.",
    "Zhang Weili became China's first-ever UFC champion when she defeated Jessica Andrade for the Strawweight title in 2019.",
    "The fastest knockout in UFC history occurred in just 5 seconds, when Jorge Masvidal landed a flying knee against Ben Askren at UFC 239.",
    "The Ultimate Fighter (TUF) is a long-running reality television show and MMA competition that launched in 2005, credited with saving the UFC from financial ruin.",
    "Alexander Volkanovski defended his Featherweight title five times, including an iconic trilogy sweep against former champion Max Holloway."
];
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

sample_query = "Who did Alex Volkanovski fight 3 times?";
sample_vect = embeddings_model.embed_query(sample_query);
print(len(sample_vect), sample_vect[0:5]);

QUESTION = "Who submitted the guy with the most submissions in UFC History?";

for chunk, score in vector_db.similarity_search_with_score(QUESTION, k=3) :
    print(f"Score: {score:.2f}");
    print(chunk.page_content);
    print(chunk.metadata);
    print("\n");

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



print("Answer with RAG: ", answer(QUESTION, vector_db));




