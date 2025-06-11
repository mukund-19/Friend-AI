from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime
import os
import torch
import numpy as np
import faiss
import pickle
import time
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
from pymongo import MongoClient

MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["friend_ai_db"]
users_collection = db["users"]
data_collection = db["friend_data"]

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Global variables for models
sentence_transformer = None
tokenizer = None
model = None
loaded_models = False

# Memory optimization flags
models_loaded_time = 0
MODEL_UNLOAD_AFTER_SECONDS = 300  # Unload models after 5 minutes of inactivity

# Pydantic models
class UploadData(BaseModel):
    friend_data: str

class ChatMessage(BaseModel):
    user_id: str
    message: str

# Load models on demand to save memory
def load_models():
    global sentence_transformer, tokenizer, model, loaded_models, models_loaded_time
    
    if loaded_models:
        models_loaded_time = time.time()
        return
    
    logger.info("Loading models...")
    
    # Load sentence transformer
    from sentence_transformers import SentenceTransformer
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load GPT-2 tokenizer and model
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    loaded_models = True
    models_loaded_time = time.time()
    logger.info("Models loaded successfully")

# Unload models to save memory
def unload_models():
    global sentence_transformer, tokenizer, model, loaded_models
    
    if not loaded_models:
        return
    
    logger.info("Unloading models to save memory...")
    
    # Delete model objects
    del sentence_transformer
    del tokenizer
    del model
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    sentence_transformer = None
    tokenizer = None
    model = None
    loaded_models = False
    logger.info("Models unloaded")

# Check if models should be unloaded due to inactivity
def check_unload_models():
    global models_loaded_time, loaded_models
    
    if loaded_models and (time.time() - models_loaded_time > MODEL_UNLOAD_AFTER_SECONDS):
        unload_models()

# Database functions
def create_user(user_id):
    user_data = {
        "user_id": user_id,
        "created_at": datetime.now()
    }
    result = users_collection.insert_one(user_data)
    return str(result.inserted_id)

def store_friend_data(user_id, raw_text):
    data = {
        "user_id": user_id,
        "raw_text": raw_text,
        "created_at": datetime.now()
    }
    result = data_collection.insert_one(data)
    return str(result.inserted_id)

def get_friend_data(user_id):
    cursor = data_collection.find({"user_id": user_id})
    return [doc["raw_text"] for doc in cursor]

# Embedding functions
def create_embeddings(user_id, text):
    # Ensure models are loaded
    load_models()
    
    # Split text into sentences or chunks
    sentences = text.split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    
    # Create embeddings
    embeddings = sentence_transformer.encode(sentences)
    
    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index and sentences
    faiss.write_index(index, f"embeddings/{user_id}_index.faiss")
    with open(f"embeddings/{user_id}_sentences.pkl", "wb") as f:
        pickle.dump(sentences, f)
    
    return len(sentences)

def get_similar_context(user_id, query, k=3):
    # Ensure models are loaded
    load_models()
    
    # Load index and sentences
    try:
        index = faiss.read_index(f"embeddings/{user_id}_index.faiss")
        with open(f"embeddings/{user_id}_sentences.pkl", "rb") as f:
            sentences = pickle.load(f)
        
        # Create query embedding
        query_embedding = sentence_transformer.encode([query])
        
        # Search for similar sentences
        D, I = index.search(np.array(query_embedding).astype('float32'), k)
        
        # Return similar sentences as context
        context = " ".join([sentences[i] for i in I[0] if i < len(sentences)])
        return context
    except Exception as e:
        logger.error(f"Error getting similar context: {e}")
        return ""

# Generation function
def generate_response(context, user_input):
    # Ensure models are loaded
    load_models()
    
    # Create prompt with context and user input
    prompt = context + "\nHuman: " + user_input + "\nFriend:"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate with memory-efficient settings
    with torch.no_grad():  # Disable gradient calculation to save memory
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 50,  # Limit output length
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the response part
    response = response.replace(prompt, "").strip()
    
    return response

# API endpoints
@app.get("/")
async def root():
    # Check if models should be unloaded
    check_unload_models()
    return {"status": "Friend AI API is running"}

@app.post("/api/upload_data")
async def upload_data(data: UploadData, background_tasks: BackgroundTasks):
    try:
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        
        # Store raw data in MongoDB
        create_user(user_id)
        store_friend_data(user_id, data.friend_data)
        
        # Create embeddings in the background to avoid timeout
        background_tasks.add_task(create_embeddings, user_id, data.friend_data)
        
        return {
            "status": "success",
            "user_id": user_id,
            "message": "Data uploaded and processing started"
        }
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    try:
        # Get context based on user message
        context = get_similar_context(message.user_id, message.message)
        
        # Generate response
        response = generate_response(context, message.message)
        
        # Schedule model unloading check
        background_tasks.add_task(check_unload_models)
        
        return {
            "response": response
        }
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Friend AI API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Friend AI API")
    unload_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
