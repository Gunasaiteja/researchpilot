"""Document ingestion: load -> chunk -> embed -> store."""
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .vectorstore import get_store

ALLOWED = {".pdf", ".txt", ".md"}


def ingest_file(path: str, original_name: str) -> dict:
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = PyPDFLoader(path) if ext == ".pdf" else TextLoader(path, encoding="utf-8")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    doc_id = str(uuid.uuid4())
    for c in chunks:
        c.metadata["doc_id"] = doc_id
        c.metadata["source"] = original_name

    get_store().add_documents(chunks)
    return {"doc_id": doc_id, "filename": original_name, "chunks": len(chunks)}
