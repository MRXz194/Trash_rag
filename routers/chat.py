from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import get_db
from models.models import Chat, ChatMessage
from models.schema import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    MessageCreate,
    MessageResponse,
    MessageListResponse,
    RAGResponse,
)
from services.rag_service import chat_with_rag

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.post("", response_model=ChatResponse)
def create_chat(data: ChatCreate, db: Session = Depends(get_db)):
    chat = Chat(title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return ChatResponse(id=chat.id, title=chat.title, created_at=chat.created_at)


@router.get("", response_model=ChatListResponse)
def list_chats(db: Session = Depends(get_db)):
    stmt = select(Chat).order_by(Chat.created_at.desc())
    result = db.execute(stmt)
    chats = result.scalars().all()
    return ChatListResponse(
        chats=[
            ChatResponse(id=c.id, title=c.title, created_at=c.created_at) for c in chats
        ]
    )


@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Ko tìm thấy chat")
    return ChatResponse(id=chat.id, title=chat.title, created_at=chat.created_at)


@router.post("/{chat_id}/message", response_model=RAGResponse)
def send_message(chat_id: int, data: MessageCreate, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Ko tìm thấy chat")

    assistant_message, sources = chat_with_rag(data.content, chat_id, db)

    db.commit()

    return RAGResponse(
        answer=MessageResponse(
            id=assistant_message.id,
            chat_id=assistant_message.chat_id,
            role=assistant_message.role,
            content=assistant_message.content,
            created_at=assistant_message.created_at,
        ),
        sources=sources,
    )


@router.get("/{chat_id}/messages", response_model=MessageListResponse)
def get_messages(chat_id: int, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Ko tìm thấy chat")

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at)
    )
    result = db.execute(stmt)
    messages = result.scalars().all()

    return MessageListResponse(
        messages=[
            MessageResponse(
                id=m.id,
                chat_id=m.chat_id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ]
    )


@router.delete("/{chat_id}")
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Ko tìm thấy chat")

    db.delete(chat)
    db.commit()
    return {"message": "Xóa chat thành công", "id": chat_id}
