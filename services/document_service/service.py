import os
import re
import uuid
import unicodedata
import logging
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from openai import OpenAI

from config import get_settings
from models import Document, DocumentChunk

logger = logging.getLogger(__name__)

settings = get_settings()

openai_client = OpenAI(api_key=settings.openai_api_key)

vietocr_predictor = None
docling_converter = None
docling_chunker = None


def get_vietocr_predictor() -> Predictor:
    global vietocr_predictor
    if vietocr_predictor is None:
        logger.info("Loading VietOCR model (one-time)...")
        config = Cfg.load_config_from_name("vgg_transformer")
        config["cnn"]["pretrained"] = True
        config["device"] = "cpu"
        vietocr_predictor = Predictor(config)
        logger.info("VietOCR model loaded")
    return vietocr_predictor


def get_docling_converter() -> DocumentConverter:
    global docling_converter
    if docling_converter is None:

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False

        docling_converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    return docling_converter


def get_docling_chunker() -> HybridChunker:
    global docling_chunker
    if docling_chunker is None:
        docling_chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=512,
            merge_peers=True,
        )
    return docling_chunker


IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "doc", "pptx", "xlsx", "html", "md", "txt"}


def ensure_upload_dir() -> Path:
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def split_text_into_chunks(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?。\n])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=settings.embedding_model,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai_client.embeddings.create(
            input=batch,
            model=settings.embedding_model,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def process_document(file_path: str) -> list[dict]:
    converter = get_docling_converter()
    result = converter.convert(file_path)
    doc = result.document

    chunker = get_docling_chunker()
    chunks = list(chunker.chunk(doc))

    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        chunk_text = clean_text(chunk_text)

        if not chunk_text:
            continue

        meta = {}
        if hasattr(chunk, "meta"):
            meta = {
                "headings": chunk.meta.headings if hasattr(chunk.meta, "headings") else [],
                "page": chunk.meta.doc_items[0].prov[0].page_no
                if hasattr(chunk.meta, "doc_items")
                and chunk.meta.doc_items
                and chunk.meta.doc_items[0].prov
                else None,
            }
        chunk_data.append({"text": chunk_text, "metadata": meta, "index": i})

    return chunk_data


def run_vietocr(image_path: str) -> str:
    predictor = get_vietocr_predictor()
    img = Image.open(image_path)
    text = predictor.predict(img)
    return text


def process_image(file_path: str) -> list[dict]:
    text = run_vietocr(file_path)
    text = clean_text(text)

    if not text:
        return []

    text_chunks = split_text_into_chunks(text, max_chars=500, overlap=50)

    chunk_data = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_data.append({
            "text": chunk_text,
            "metadata": {"source": "vietocr", "file": file_path, "part": i + 1, "total_parts": len(text_chunks)},
            "index": i,
        })

    return chunk_data


def save_uploaded_files(files: list[tuple[str, str, str | None]], chat_id: int, db: Session) -> list[Document]:
    upload_dir = ensure_upload_dir()
    documents = []

    for filename, file_content_path, content_type in files:
        file_ext = os.path.splitext(filename)[1].lower()
        stored_filename = f"{uuid.uuid4()}{file_ext}"
        stored_path = upload_dir / stored_filename

        with open(file_content_path, "rb") as src, open(stored_path, "wb") as dst:
            dst.write(src.read())

        doc = Document(
            filename=filename,
            content_type=content_type,
            chat_id=chat_id,
            processing_status="pending",
            stored_path=str(stored_path),
        )
        db.add(doc)
        db.flush()
        documents.append(doc)

    return documents
