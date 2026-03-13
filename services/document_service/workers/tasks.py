import hashlib
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from workers.celery_app import celery_app
from database import SessionLocal
from models import Document, DocumentChunk
from service import (
    process_document,
    process_image,
    get_embeddings_batch,
    IMAGE_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def compute_chunk_hash(document_id: int, text: str) -> str:
    content = f"{document_id}:{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@celery_app.task(bind=True, name="workers.tasks.process_document_task", max_retries=2)
def process_document_task(self, document_id: int):
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if not doc:
            logger.error(f"Document {document_id} not found")
            return {"status": "failed", "error": "Document not found"}

        doc.processing_status = "processing"
        db.commit()

        file_path = doc.stored_path
        file_ext = doc.filename.rsplit(".", 1)[-1].lower() if "." in doc.filename else ""

        chunks_data = []

        if file_ext in IMAGE_EXTENSIONS:
            try:
                chunks_data = process_image(file_path)
            except Exception as e:
                logger.warning(f"VietOCR failed for {doc.filename}: {e}")
                chunks_data = [{
                    "text": f"[Image: {doc.filename} - could not extract text]",
                    "metadata": {"error": str(e)},
                    "index": 0,
                }]

        elif file_ext in DOCUMENT_EXTENSIONS:
            try:
                chunks_data = process_document(file_path)
            except Exception as e:
                logger.warning(f"Docling failed for {doc.filename}: {e}")
                chunks_data = [{
                    "text": f"[Document: {doc.filename} - could not extract text]",
                    "metadata": {"error": str(e)},
                    "index": 0,
                }]

        else:
            logger.warning(f"Unsupported file type: {file_ext} for {doc.filename}")
            chunks_data = [{
                "text": f"[Unsupported: {doc.filename} - file type '{file_ext}' not supported]",
                "metadata": {"error": f"Unsupported file type: {file_ext}"},
                "index": 0,
            }]

        if chunks_data:
            texts = [c["text"] for c in chunks_data]
            embeddings = get_embeddings_batch(texts)

            created = 0
            skipped = 0
            for chunk, embedding in zip(chunks_data, embeddings):
                chunk_hash = compute_chunk_hash(doc.id, chunk["text"])

                existing = db.query(DocumentChunk).filter_by(chunk_hash=chunk_hash).first()
                if existing:
                    skipped += 1
                    continue

                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    text=chunk["text"],
                    embedding=embedding,
                    chunk_metadata=chunk.get("metadata", {}),
                    chunk_index=chunk["index"],
                    chunk_hash=chunk_hash,
                )
                db.add(db_chunk)
                created += 1

        doc.processing_status = "completed"
        db.commit()

        logger.info(f"Document {document_id} processed: {created} created, {skipped} duplicates skipped")
        return {"status": "completed", "chunks_created": created, "duplicates_skipped": skipped}

    except Exception as e:
        db.rollback()
        logger.error(f"Document {document_id} processing failed: {e}")

        try:
            doc = db.get(Document, document_id)
            if doc:
                doc.processing_status = "failed"
                doc.processing_error = str(e)[:500]
                db.commit()
        except Exception:
            db.rollback()

        raise self.retry(exc=e, countdown=30)

    finally:
        db.close()
