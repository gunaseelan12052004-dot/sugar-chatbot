import os
from typing import List
import torch
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, preserve only page_content and source metadata.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split text into chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

class MiniLMEmbeddings(Embeddings):
    """
    Fast and reliable SentenceTransformer embedding wrapper for LangChain.
    Prefers locally cached weights to avoid network timeouts.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

def download_embedding():
    return MiniLMEmbeddings()
