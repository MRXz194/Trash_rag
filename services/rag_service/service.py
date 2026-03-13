import json
import logging
import httpx
import redis
from openai import OpenAI

from config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
http_client = httpx.Client(timeout=30.0)
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

HISTORY_KEY = "chat:{chat_id}:messages"


def get_chat_history(chat_id: int) -> list[dict]:
    key = HISTORY_KEY.format(chat_id=chat_id)
    try:
        raw_messages = redis_client.lrange(key, 0, -1)
        if raw_messages:
            messages = [json.loads(m) for m in raw_messages]
            return messages[-settings.max_history_messages:]
        return load_history_from_db(chat_id)
    except redis.RedisError as e:
        logger.warning(f"Redis thất bại, fallback to DB: {e}")
        return load_history_from_db(chat_id)


def load_history_from_db(chat_id: int) -> list[dict]:
    from database import SessionLocal
    from models import ChatMessage
    from sqlalchemy import select

    db = SessionLocal()
    try:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
        )
        result = db.execute(stmt)
        messages = result.scalars().all()

        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        cache_history_to_redis(chat_id, history)

        return history[-settings.max_history_messages:]
    finally:
        db.close()


def cache_history_to_redis(chat_id: int, messages: list[dict]):
    key = HISTORY_KEY.format(chat_id=chat_id)
    try:
        pipe = redis_client.pipeline()
        pipe.delete(key)
        for msg in messages[-settings.max_history_messages:]:
            pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
        pipe.expire(key, settings.history_ttl_seconds)
        pipe.execute()
    except redis.RedisError as e:
        logger.warning(f"Redis cache write failed: {e}")


def save_message_to_redis(chat_id: int, role: str, content: str):
    key = HISTORY_KEY.format(chat_id=chat_id)
    msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    try:
        pipe = redis_client.pipeline()
        pipe.rpush(key, msg)
        pipe.ltrim(key, -settings.max_history_messages, -1)
        pipe.expire(key, settings.history_ttl_seconds)
        pipe.execute()
    except redis.RedisError as e:
        logger.warning(f"Redis save failed: {e}")


def rewrite_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    recent = history[-6:]

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    )

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriter. Given the conversation history and the latest user query, "
                        "rewrite the query to be self-contained and clear, resolving any pronouns or references. "
                        "Return ONLY the rewritten query, nothing else. "
                        "If the query is already clear and self-contained, return it as-is. "
                        "Keep the same language as the original query."
                    ),
                },
                {
                    "role": "user",
                    "content": f"CONVERSATION HISTORY:\n{history_text}\n\nLATEST QUERY: {query}\n\nREWRITTEN QUERY:",
                },
            ],
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")

    return query


def search_relevant_chunks(query: str, chat_id: int, top_k: int | None = None) -> list[dict]:
    if top_k is None:
        top_k = settings.retrieval_top_k

    fetch_k = top_k * settings.rerank_multiplier

    try:
        response = http_client.post(
            f"{settings.document_service_url}/internal/search",
            json={"query": query, "chat_id": chat_id, "top_k": fetch_k},
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("chunks", [])
    except httpx.RequestError as e:
        logger.error(f"Document Service unreachable: {e}")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"Document Service error: {e.response.text}")
        return []

    if len(candidates) <= top_k:
        return candidates

    return rerank_chunks(query, candidates, top_k)


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"][:300]
        chunk_summaries.append(f"[{i}] {text}")

    chunks_text = "\n---\n".join(chunk_summaries)

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance scoring assistant. Given a query and a list of text chunks, "
                        "rate each chunk's relevance to the query on a scale of 0-10. "
                        "Return ONLY a comma-separated list of scores in the same order as the chunks. "
                        "Example output for 5 chunks: 8,3,9,1,6"
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUERY: {query}\n\nCHUNKS:\n{chunks_text}\n\nSCORES:",
                },
            ],
        )
        scores_text = response.choices[0].message.content.strip()
        scores = [int(s.strip()) for s in scores_text.split(",")]

        if len(scores) != len(chunks):
            logger.warning(f"Re-rank score count mismatch: {len(scores)} vs {len(chunks)}, using original order")
            return chunks[:top_k]

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        reranked = [chunk for _, chunk in scored_chunks[:top_k]]
        logger.info(f"Re-ranked {len(chunks)} chunks -> top {top_k}, scores: {scores}")
        return reranked

    except Exception as e:
        logger.warning(f"Re-ranking failed, using original order: {e}")
        return chunks[:top_k]


def build_rag_prompt(context_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        source_info = ""
        if meta.get("headings"):
            source_info = f" (Section: {' > '.join(meta['headings'])})"
        if meta.get("page"):
            source_info += f" (Page {meta['page']})"
        context_parts.append(f"[{i}]{source_info}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert data analyst and research assistant. Your primary directive is to provide accurate, comprehensive, and well-structured answers based EXCLUSIVELY on the provided Document Context.

====================
DOCUMENT CONTEXT:
{context}
====================

STRICT OPERATING RULES:
1. NO HALLUCINATION: You must rely entirely on the DOCUMENT CONTEXT. Do not incorporate prior training knowledge, assumptions, or external facts.
2. HANDLING MISSING INFO: If the DOCUMENT CONTEXT does not contain the necessary information to fully or partially answer the prompt, explicitly state: "The provided documents do not contain sufficient information to answer this query."
3. MANDATORY CITATIONS: Every claim, fact, or metric you provide MUST be backed by a citation from the DOCUMENT CONTEXT. Use inline citations referencing the source name, metadata, or paragraph if available (e.g., "According to [Document Name]...").
4. FORMATTING: Structure your response for maximum readability. Use markdown, bullet points, numbered lists, or tables where appropriate to break down complex information.
5. NEUTRAL TONE: Maintain a professional, objective, and unbiased tone.
6. Answer the same language as the user query.
"""


def generate_rag_response(query: str, chat_id: int) -> tuple[str, list[dict]]:
    history = get_chat_history(chat_id)

    rewritten_query = rewrite_query(query, history)

    chunks = search_relevant_chunks(rewritten_query, chat_id)

    if not chunks:
        return (
            "Ko có tài liệu liên quan. Up file lên trước.",
            [],
        )

    system_prompt = build_rag_prompt(chunks)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-settings.max_history_messages:])
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return (
            "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
            [],
        )

    sources = []
    for chunk in chunks:
        text = chunk["text"]
        sources.append(
            {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "metadata": chunk.get("metadata", {}),
                "document_id": chunk.get("document_id"),
            }
        )

    return answer, sources