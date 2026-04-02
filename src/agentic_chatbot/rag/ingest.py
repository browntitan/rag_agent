from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from agentic_chatbot.config import Settings
from agentic_chatbot.db.chunk_store import ChunkRecord
from agentic_chatbot.db.document_store import DocumentRecord
from agentic_chatbot.rag.clause_splitter import clause_split
from agentic_chatbot.rag.ocr import IMAGE_SUFFIXES, load_image_documents, load_pdf_documents_with_ocr
from agentic_chatbot.rag.stores import KnowledgeStores, make_doc_id
from agentic_chatbot.rag.structure_detector import (
    REQUIREMENT_PATTERN,
    StructureAnalysis,
    detect_structure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_documents(path: Path, settings: Settings) -> List[Document]:
    """Load a file into LangChain Documents.

    Dispatch:
      .md / .txt       → TextLoader (unchanged)
      .pdf             → hybrid PyPDF + PaddleOCR when ocr_enabled, else PyPDFLoader
      .docx            → Docx2txtLoader (unchanged)
      image suffixes   → PaddleOCR when ocr_enabled, else skip (return [])
      other            → TextLoader with autodetect_encoding (fallback)
    """
    suffix = path.suffix.lower()

    if suffix in {".md", ".txt"}:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(path), encoding="utf-8").load()

    if suffix == ".pdf":
        if settings.ocr_enabled:
            return load_pdf_documents_with_ocr(
                path,
                min_page_chars=settings.ocr_min_page_chars,
                language=settings.ocr_language,
                use_gpu=settings.ocr_use_gpu,
            )
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(path)).load()

    if suffix == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(str(path)).load()
        except Exception:
            pass  # fall through to text fallback

    if suffix in IMAGE_SUFFIXES:
        if settings.ocr_enabled:
            return load_image_documents(
                path,
                language=settings.ocr_language,
                use_gpu=settings.ocr_use_gpu,
            )
        logger.debug("OCR disabled; skipping image file %s.", path)
        return []

    # Generic text fallback
    from langchain_community.document_loaders import TextLoader
    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


# ---------------------------------------------------------------------------
# Splitting: structure-aware dispatch
# ---------------------------------------------------------------------------

def _general_split(settings: Settings, docs: List[Document]) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def _split_with_structure(
    settings: Settings,
    docs: List[Document],
    structure: StructureAnalysis,
) -> List[Document]:
    """Choose clause-aware or generic splitting based on detected structure.

    In both paths, chunks are post-processed to tag chunk_type='requirement'
    wherever requirement language is detected in a 'general' chunk.
    """
    if structure.has_clauses:
        chunks: List[Document] = []
        for doc in docs:
            chunks.extend(
                clause_split(
                    doc,
                    max_clause_chars=settings.chunk_size * 2,
                    overlap_chars=settings.chunk_overlap,
                )
            )
    else:
        chunks = _general_split(settings, docs)
        # Tag requirement chunks in general docs
        for ch in chunks:
            if REQUIREMENT_PATTERN.search(ch.page_content):
                ch.metadata["chunk_type"] = "requirement"
            elif not ch.metadata.get("chunk_type"):
                ch.metadata["chunk_type"] = "general"

    return chunks


# ---------------------------------------------------------------------------
# ChunkRecord builder
# ---------------------------------------------------------------------------

def _build_chunk_records(
    chunks: List[Document],
    doc_id: str,
    *,
    collection_id: str,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        meta = ch.metadata or {}
        # Assign sequential index if the splitter didn't set one
        chunk_index = int(meta.get("chunk_index", i))
        chunk_id = f"{doc_id}#chunk{chunk_index:04d}"

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                collection_id=collection_id,
                chunk_index=chunk_index,
                content=ch.page_content,
                chunk_type=str(meta.get("chunk_type", "general")),
                page_number=meta.get("page"),
                clause_number=meta.get("clause_number") or None,
                section_title=meta.get("section_title") or None,
                embedding=None,  # ChunkStore will embed on insert
            )
        )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_paths(
    settings: Settings,
    stores: KnowledgeStores,
    paths: Iterable[Path],
    *,
    source_type: str,
    tenant_id: str,
    collection_id: str | None = None,
) -> List[str]:
    """Ingest files into PostgreSQL (chunks + documents tables).

    Skips files that have already been ingested with the same content hash.
    Returns the list of newly ingested doc_ids.
    """
    if settings.object_store_backend != "local":
        raise NotImplementedError(
            f"OBJECT_STORE_BACKEND={settings.object_store_backend!r} is not implemented for ingest yet. "
            "Set OBJECT_STORE_BACKEND=local for now."
        )

    ingested_doc_ids: List[str] = []
    effective_collection_id = collection_id or settings.default_collection_id

    for p in paths:
        p = Path(p)
        if not p.exists() or not p.is_file():
            continue

        file_hash = _file_hash(p)
        title = p.name
        doc_id = make_doc_id(
            source_type=source_type,
            title=title,
            content_hash=file_hash,
            tenant_id=tenant_id,
        )

        # Skip if already ingested with identical content
        if stores.doc_store.document_exists(doc_id, file_hash, tenant_id=tenant_id):
            continue

        raw_docs = _load_documents(p, settings)
        if not raw_docs:
            logger.info("No content extracted from %s — skipping.", p)
            continue

        # Attach base metadata to raw docs before splitting
        for d in raw_docs:
            d.metadata = {
                **(d.metadata or {}),
                "doc_id": doc_id,
                "title": title,
                "source_type": source_type,
                "source_path": str(p),
            }

        full_text = " ".join(d.page_content for d in raw_docs)
        structure = detect_structure(full_text)

        chunks = _split_with_structure(settings, raw_docs, structure)
        chunk_records = _build_chunk_records(chunks, doc_id, collection_id=effective_collection_id)

        # Persist parent doc row first (chunks.doc_id has FK -> documents.doc_id).
        stores.doc_store.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
                collection_id=effective_collection_id,
                title=title,
                source_type=source_type,
                content_hash=file_hash,
                source_path=str(p),
                num_chunks=len(chunk_records),
                ingested_at=dt.datetime.utcnow().isoformat() + "Z",
                file_type=p.suffix.lstrip(".").lower(),
                doc_structure_type=structure.doc_structure_type,
            )
        )

        # Persist chunks (embeddings computed inside ChunkStore.add_chunks).
        # If this fails, remove the document row to avoid orphan metadata.
        try:
            stores.chunk_store.add_chunks(chunk_records, tenant_id=tenant_id)
        except Exception:
            stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
            raise

        ingested_doc_ids.append(doc_id)

    return ingested_doc_ids


def ensure_kb_indexed(settings: Settings, stores: KnowledgeStores, tenant_id: str) -> None:
    """Index the built-in KB documents if the documents table is empty for source_type='kb'."""
    if not settings.seed_demo_kb_on_startup:
        return
    if settings.object_store_backend != "local":
        raise NotImplementedError(
            f"OBJECT_STORE_BACKEND={settings.object_store_backend!r} is not implemented for KB indexing yet. "
            "Set OBJECT_STORE_BACKEND=local for now."
        )
    kb_docs = stores.doc_store.list_documents(
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=settings.default_collection_id,
    )
    if kb_docs:
        return
    kb_paths = sorted(Path(settings.kb_dir).glob("*"))
    ingest_paths(
        settings,
        stores,
        kb_paths,
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=settings.default_collection_id,
    )
