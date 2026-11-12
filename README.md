# DocuMind AI - RAG Document Chatbot

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application built with Python.

Upload PDF documents, ask questions about them, and get answers grounded in the uploaded document with page sources.

## Features
- Upload multiple PDFs at once (Multi-PDF Support)
- Upload one or more PDF documents
- Extract text using `pypdf`
- Split document text into overlapping chunks
- Generate embeddings using Sentence Transformers
- Store and search embeddings with FAISS
- Retrieve the most relevant document sections for each question
- Generate grounded answers using Groq
- Chat-style interface using Streamlit
- Maintain conversation history with Streamlit session state
- Show page-level sources for every answer
- Clear chat history when a different PDF is uploaded
- Handle PDFs with no readable text
- Handle missing API keys and model/API errors
- Avoid answering from outside the uploaded document

## Tech Stack

- Python
- Streamlit
- pypdf
- Sentence Transformers
- FAISS
- Groq
- python-dotenv

## How RAG Works in This Project

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Document Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
Groq LLM
    ↓
Grounded Answer + Page Sources



Project Structure

DocuMind-AI/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

Setup

1. Create a virtual environment
        python -m venv .venv
        source .venv/bin/activate

2. Install dependencies
        python -m pip install -r requirements.txt

Environment Variables
    Create a .env file:
        GROQ_API_KEY=your_actual_groq_api_key
        GROQ_MODEL=openai/gpt-oss-20b


Run the Application:
        streamlit run app.py


How to Use

 1.  Upload a text-based PDF.
 2.  Wait for the document to be processed.
 3.  Enter a question in the chat box.
 4.  The app retrieves the most relevant sections using FAISS.
 5.  Groq generates an answer using only the retrieved context.
 6.  Continue asking questions from the same document.


 Sample Test (my project report):

Tested with Pothole Detection Project Report (83 pages).

Example questions:
        What is the scope of this project?
        What techniques are used for pothole detection?

    And it will search and give you the answer which is located in your uploaded pdf..

Test grounding: 
Who is the CEO of the company?
    The chatbot should say it could not find it instead of inventing an answer.

RAG Components Used

1. Text Extraction
pypdf reads the PDF and extracts text from each page.

2. Chunking
    Chunk size: 180 words
    Chunk overlap: 40 words
Overlap helps preserve context near chunk boundaries.

3. Embeddings
    sentence-transformers/all-MiniLM-L6-v2

4. Vector Search
FAISS stores embeddings and retrieves top 3 most relevant chunks for each question.

5. Answer Generation
Retrieved chunks are sent to Groq. The model is instructed to:
 •  Answer only using supplied context
 •  If exact answer is not there, use closest related context and say so
 •  Keep answers clear and concise


Learning Outcomes

 •  What RAG is
 •  Why documents are split into chunks
 •  What embeddings represent
 •  How vector similarity search works
 •  How FAISS retrieves context
 •  How to build an AI chat interface with Streamlit


Author
        Varshanth - Aspiring AI Developer

License
This project is intended for learning and demonstration purposes.