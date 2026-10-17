from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from pathlib import Path

docs_by_hand = [
  Document(page_content="Python is a programming language",
           metadata={"source": "intro.txt", "level": "beginner"}),
  Document(page_content="Embeddings turn text into vectors",
           metadata={"source": "intro1.txt", "level": "beginner"})
];

print([d.page_content for d in docs_by_hand]);

PDFs = Path("utils");
pdf_path = sorted(PDFs.glob("*.pdf"))[0];
print(pdf_path);

loader = PyPDFLoader(str(pdf_path));
pages = loader.load();
# print(type(pages));
# print(len(pages));
# print(pages[0]);

for i, page in enumerate(pages) :
    print(f"Page {i+1}");
    print(page.page_content);
    print("\n");
