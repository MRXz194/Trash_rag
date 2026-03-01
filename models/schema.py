from pydantic import BaseModel
from datetime import datetime


class ChatCreate(BaseModel):
    title: str


class ChatResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    chats: list[ChatResponse]


class MessageCreate(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: int
    chat_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str | None
    chat_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentResponse]
    chat_id: int
    chunks_created: int


class RAGResponse(BaseModel):
    answer: MessageResponse
    sources: list[dict]
