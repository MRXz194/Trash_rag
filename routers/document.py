import os
import tempfile

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import get_db
from models.models import Document, DocumentChunk, Chat
from models.schema import DocumentResponse, DocumentListResponse, DocumentUploadResponse
from services.document_service import process_and_store_documents

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_documents(files: list[UploadFile] = File(...), chat_id: int | None = Form(None), chat_title: str = Form("New Chat"), db: Session = Depends(get_db),):
    if not files:
        raise HTTPException(status_code=400, detail="Ko tìm thấy file")

    if chat_id:
        chat = db.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Ko tìm thấy chat")
    else:
        chat = Chat(title=chat_title)
        db.add(chat)
        db.flush()
        chat_id = chat.id

    file_infos = []
    temp_paths = []

    try:
        for file in files:
            suffix = os.path.splitext(file.filename or "unknown")[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = file.file.read()
            tmp.write(content)
            tmp.close()
            temp_paths.append(tmp.name)
            file_infos.append((file.filename or "unknown", tmp.name, file.content_type))

        documents, chunks_count = process_and_store_documents(file_infos, chat_id, db)

        db.commit()

        doc_responses = [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                chat_id=doc.chat_id,
                created_at=doc.created_at,
            )
            for doc in documents
        ]

        return DocumentUploadResponse(
            documents=doc_responses,
            chat_id=chat_id,
            chunks_created=chunks_count,
        )

    except Exception:
        db.rollback()
        raise
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


@router.get("/{chat_id}", response_model=DocumentListResponse)
def list_documents(chat_id: int, db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Ko tìm thấy chat")

    stmt = select(Document).where(Document.chat_id == chat_id).order_by(Document.created_at)
    result = db.execute(stmt)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                content_type=doc.content_type,
                chat_id=doc.chat_id,
                created_at=doc.created_at,
            )
            for doc in documents
        ]
    )


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Ko tìm thấy doc")

    db.delete(doc)
    db.commit()
    return {"message": "Xóa doc thành công", "id": document_id}
