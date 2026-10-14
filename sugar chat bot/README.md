# 🩺 Medical Sugar (Diabetes) Chatbot

An AI-powered clinical question-answering assistant specializing in **Diabetes Mellitus and Blood Sugar Health**, built using **RAG (Retrieval-Augmented Generation)**, **Pinecone**, **Sentence Transformers**, **Groq LPU**, and **Flask**.

---

## 🚀 Live Demo & Quick Start

### 1. Run the Web Application
```powershell
python app.py
```
Open your browser at: **`http://127.0.0.1:5000`**

### 2. Ingest New Documents (Optional)
If you add new PDFs to the `data/` folder, run:
```powershell
python store_index.py
```

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.13
- **Framework:** Flask 3.1.1
- **Orchestration:** LangChain 0.3.26
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database:** Pinecone Cloud (Serverless AWS)
- **Generative LLM:** `qwen/qwen3.8-27b` via Groq High-Speed LPU
- **Document Processing:** PyPDF & DirectoryLoader
- **Similarity Metric:** Cosine Similarity ($k=3$)
- **Algorithm:** Retrieval-Augmented Generation (RAG)

---

## 📁 Detailed Technical Documentation

For the complete technical breakdown including mathematical formulations, model comparison, and architectural workflows, see:
👉 **[PROJECT_DETAILS.Rmd](PROJECT_DETAILS.Rmd)**
