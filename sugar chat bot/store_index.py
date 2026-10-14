import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embedding

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-sugar-chatbot")

print(f"Loading data from 'data' directory...")
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)
print(f"Total chunks created: {len(texts_chunk)}")

print("Loading embedding model...")
embeddings = download_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully!")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

print("Upserting documents to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("Data successfully uploaded to Pinecone! ✅")
