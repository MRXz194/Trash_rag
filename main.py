import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine, Base
from routers import document, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.models import Document, DocumentChunk, Chat, ChatMessage

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(bind=engine)
logger.info("Database tables created successfully")

app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {"message": "RAG Service is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}
