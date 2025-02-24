from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import fitz  # PyMuPDF
from collections import defaultdict, deque
import redis
import torch
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from serpapi import GoogleSearch

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index Setup
dimension = 384  # MiniLM has 384 dimensions
index = faiss.IndexFlatL2(dimension)

stored_texts = []  # To map embeddings to text

def store_embeddings(text_chunks):
    """Convert text to embeddings and store in FAISS index."""
    global stored_texts
    embeddings = embedding_model.encode(text_chunks)
    index.add(np.array(embeddings, dtype=np.float32))
    stored_texts.extend(text_chunks)

def retrieve_relevant_text(query):
    """Find the most relevant text chunks for a given query."""
    if index.ntotal == 0 or len(indices[0]) == 0:        # Check if FAISS index is empty
        return ""

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=3)

    if indices is None or len(indices) == 0 or len(indices[0]) == 0:
        return ""

    return " ".join([stored_texts[i] for i in indices[0] if i < len(stored_texts)])
    retrieved_text = retrieve_relevant_text(data.question)
    print("Retrieved Text:", retrieved_text)  # Debugging line



SERPAPI_KEY = os.getenv("SERPAPI_KEY")  


def web_search(query):
    """Fetch relevant information from Google Search."""
    if not SERPAPI_KEY:
        return "No additional information available."
    
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3
    })
    
    results = search.get_dict().get("organic_results", [])
    return " ".join([res["snippet"] for res in results])

app = FastAPI()

# Load NLP Model with optimized settings
device = 0 if torch.cuda.is_available() else -1  
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

# Chat Memory Setup (Redis or In-Memory)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  
    chat_memory = None  
except:
    redis_client = None
    chat_memory = defaultdict(lambda: deque(maxlen=10))  # Fallback to in-memory storage

class Question(BaseModel):
    question: str
    user_id: str
    context: str

import json

def store_chat(user_id, question, answer):
    """Store chat history in Redis or in-memory."""
    chat_entry = {"question": question, "answer": answer}  # Store as a dictionary
    if redis_client:
        key = f"chat:{user_id}"
        redis_client.rpush(key, json.dumps(chat_entry))  # Use JSON to store
        redis_client.ltrim(key, -10, -1)  # Keep only last 10 messages
    else:
        chat_memory[user_id].append(chat_entry)  # Store as a dictionary


def get_chat_history(user_id):
    """Retrieve chat history."""
    if redis_client:
        chat_history = redis_client.lrange(f"chat:{user_id}", 0, -1)
        # Ensure we only try to load valid JSON data
        history = []
        for entry in chat_history:
            try:
                history.append(json.loads(entry))  # Use JSON to load
            except json.JSONDecodeError:
                # Handle invalid or empty data
                print(f"Skipping invalid chat entry for user {user_id}")
                continue
        return history
    return list(chat_memory[user_id]) 

async def extract_text_from_pdf(file: UploadFile):
    """Extract text from a PDF file properly."""
    try:
        pdf_bytes = await file.read()  # Read the file into bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Load PDF from bytes
        text = " ".join(page.get_text("text") for page in doc)  # Extract text
        return [text[i:i+500] for i in range(0, len(text), 500)]  # Chunk into 500 chars
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
def ask_question(data: Question):
    """Answer user queries with memory-based context + external search."""
    user_id = data.user_id
    previous_context = " ".join(entry["answer"] for entry in get_chat_history(user_id))
    retrieved_text = retrieve_relevant_text(data.question)
    full_context = f"{previous_context} {retrieved_text}".strip()

    if not full_context.strip():
        web_data = web_search(data.question)
        if web_data:
            full_context = web_data

    try:
        response = qa_pipeline(question=data.question, context=full_context)
        if response["score"] < 0.0001:  # If confidence is low go to fetch from the web
            web_data = web_search(data.question)
            response = qa_pipeline(question=data.question, context=web_data)

        store_chat(user_id, data.question, response['answer'])
        return {"answer": response['answer'], "score": response['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")



@app.post("/ask-pdf")
async def ask_pdf(question: str = Form(...), user_id: str = Form(...), file: UploadFile = File(...)):
    """Answer questions based on uploaded PDF content."""
    pdf_chunks = await extract_text_from_pdf(file)
    previous_answers = " ".join(entry["answer"] for entry in get_chat_history(user_id)[-3:])

    best_answer = {"answer": "No relevant answer found.", "score": 0}
    for chunk in pdf_chunks:
        full_context = (previous_answers + " " + chunk)[:1024]  # Ensure no context exceeds the length
        try:
            response = qa_pipeline(question=question, context=full_context)
            if response["score"] > best_answer["score"]:
                best_answer = response
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue  # Skip on error and try next chunk

    store_chat(user_id, question, best_answer["answer"])
    return best_answer


@app.get("/history/{user_id}")
def get_chat_history_api(user_id: str):
    """Retrieve user chat history."""
    return {"chat_history": get_chat_history(user_id)}

@app.get("/health")
def health_check():
    """Check API health."""
    return {"status": "running"}

@app.get("/redis-health")
def redis_health_check():
    if redis_client:
        try:
            redis_client.ping()  # Ping Redis server
            return {"status": "Redis is working!"}
        except Exception as e:
            return {"status": f"Redis is down! Error: {str(e)}"}
    else:
        return {"status": "Redis client is not initialized."}


@app.post("/add-faq")
def add_faq(question: str, answer: str):
    """Add a custom FAQ to the system."""
    faq_entry = {"question": question, "answer": answer}
    if redis_client:
        redis_client.set(f"faq:{question}", answer)  # Store in Redis
    else:
        # Store in-memory if Redis is not available
        chat_memory["faq"].append(faq_entry)
    return {"status": "FAQ added successfully"}


@app.get("/faq/{question}")
def get_faq_answer(question: str):
    """Retrieve an answer from the custom FAQ."""
    if redis_client:
        answer = redis_client.get(f"faq:{question}")
        if answer:
            return {"answer": answer}
        else:
            return {"answer": "No answer found for this question."}
    else:
        for faq in chat_memory.get("faq", []):
            if faq["question"] == question:
                return {"answer": faq["answer"]}
        return {"answer": "No answer found for this question."}
