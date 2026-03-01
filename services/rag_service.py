import logging

from sqlalchemy.orm import Session
from sqlalchemy import select
import google.generativeai as genai

from core.config import get_settings
from models.models import DocumentChunk, ChatMessage
from services.document_service import get_embedding

logger = logging.getLogger(__name__)

settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)
gemini_model = genai.GenerativeModel(settings.gemini_model)


def retrieve_relevant_chunks(query: str, chat_id: int, db: Session, top_k: int | None = None) -> list[DocumentChunk]:
    if top_k is None:
        top_k = settings.retrieval_top_k

    query_embedding = get_embedding(query)

    stmt = (
        select(DocumentChunk)
        .join(DocumentChunk.document)
        .where(DocumentChunk.document.has(chat_id=chat_id))
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    result = db.execute(stmt)
    chunks = result.scalars().all()
    return list(chunks)


def build_rag_prompt(query: str, context_chunks: list[DocumentChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.chunk_metadata or {}
        source_info = ""
        if meta.get("headings"):
            source_info = f" (Section: {' > '.join(meta['headings'])})"
        if meta.get("page"):
            source_info += f" (Page {meta['page']})"
        context_parts.append(f"[{i}]{source_info}\n{chunk.text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the information from the context below to answer the question.
If the context doesn't contain enough information to answer, say so clearly.
When relevant, cite the source numbers in square brackets like [1], [2], etc.
Answer in the same language as the question.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    return prompt


def rag_response(query: str, chat_id: int, db: Session) -> tuple[str, list[dict]]:
    chunks = retrieve_relevant_chunks(query, chat_id, db)

    if not chunks:
        return (
            "Ko có tài liệu liên quan."
            "Up file lên trước.",
            [],
        )

    prompt = build_rag_prompt(query, chunks)

    response = gemini_model.generate_content(prompt)
    answer = response.text

    sources = []
    for chunk in chunks:
        sources.append(
            {
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "metadata": chunk.chunk_metadata or {},
                "document_id": chunk.document_id,
            }
        )

    return answer, sources


def chat_with_rag(query: str, chat_id: int, db: Session) -> tuple[ChatMessage, list[dict]]:
    user_message = ChatMessage(
        chat_id=chat_id,
        role="user",
        content=query,
    )
    db.add(user_message)
    db.flush()

    answer, sources = generate_rag_response(query, chat_id, db)

    assistant_message = ChatMessage(
        chat_id=chat_id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_message)
    db.flush()

    return assistant_message, sources
