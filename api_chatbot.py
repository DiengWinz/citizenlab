"""
API REST pour le Chatbot CitizenLab
Framework: FastAPI
Usage: Intégration sur site web existant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from chatbot_rag import ChatbotRAG

# Charger les variables d'environnement
load_dotenv()

app = FastAPI(
    title="CitizenLab Chatbot API",
    description="API REST pour le chatbot conversationnel CitizenLab",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions = {}
chatbot = None  # Lazy loading

# ----------------------------
# MODELS
# ----------------------------

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    top_k: Optional[int] = 5
    show_sources: Optional[bool] = True


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[str]] = None


# ----------------------------
# UTIL
# ----------------------------

def get_chatbot():
    global chatbot
    
    if chatbot is None:
        print("🚀 Chargement du chatbot (lazy)...")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY manquante")

        chatbot = ChatbotRAG(
            groq_api_key=groq_api_key,
            csv_folder="knowledge_base"
        )

        chatbot.load_vector_index()
        print("✅ Chatbot prêt!")

    return chatbot


# ----------------------------
# ROUTES
# ----------------------------

@app.get("/")
async def root():
    return {
        "message": "CitizenLab Chatbot API",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "running"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        bot = get_chatbot()

        if message.session_id not in chat_sessions:
            chat_sessions[message.session_id] = bot

        session_chatbot = chat_sessions[message.session_id]

        response = session_chatbot.chat(
            message.message,
            top_k=message.top_k,
            show_sources=message.show_sources
        )

        return {
            "response": response,
            "session_id": message.session_id,
            "sources": None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
