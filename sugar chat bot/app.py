import os
import sys

# Prevent TensorFlow/Keras conflicts
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_embedding
from src.prompt import system_prompt

app = Flask(__name__)

# Load embedding model
print("Loading embedding model...")
embeddings = download_embedding()

# Pinecone direct index connection
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-sugar-demo-chatbot")
print(f"Connecting to Pinecone index: '{index_name}'...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Groq LLM
print("Initializing Groq LLM...")
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.3
)

# Setup RAG chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("Medical Sugar Chatbot is ready!")


@app.route("/")
def index_route():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_msg = request.form.get("msg", "").strip()
    if not user_msg:
        user_msg = request.args.get("msg", "").strip()
    
    if not user_msg:
        return "Please ask a question."

    try:
        response = rag_chain.invoke({"input": user_msg})
        answer = response.get("answer", "I could not retrieve an answer at this time.")
        return answer
    except Exception as e:
        print(f"Error processing request: {e}")
        return "An error occurred while generating the response. Please try again."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
