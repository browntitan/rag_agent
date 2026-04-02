from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.clause_splitter import clause_split
from agentic_chatbot.rag.ocr import IMAGE_SUFFIXES, load_image_documents, load_pdf_documents_with_ocr
from agentic_chatbot.rag.structure_detector import REQUIREMENT_PATTERN, StructureAnalysis, detect_structure
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.rag.stores import KnowledgeStores, make_doc_id

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    sha = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_documents(path: Path, settings: Settings) -> List[Document]:
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
            pass

    if suffix in IMAGE_SUFFIXES:
        if settings.ocr_enabled:
            return load_image_documents(
                path,
                language=settings.ocr_language,
                use_gpu=settings.ocr_use_gpu,
            )
        logger.debug("OCR disabled; skipping image file %s.", path)
        return []

    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


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
        for chunk in chunks:
            if REQUIREMENT_PATTERN.search(chunk.page_content):
                chunk.metadata["chunk_type"] = "requirement"
            elif not chunk.metadata.get("chunk_type"):
                chunk.metadata["chunk_type"] = "general"

    return chunks


def _build_chunk_records(
    chunks: List[Document],
    doc_id: str,
    *,
    collection_id: str,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        chunk_index = int(metadata.get("chunk_index", index))
        records.append(
            ChunkRecord(
                chunk_id=f"{doc_id}#chunk{chunk_index:04d}",
                doc_id=doc_id,
                collection_id=collection_id,
                chunk_index=chunk_index,
                content=chunk.page_content,
                chunk_type=str(metadata.get("chunk_type", "general")),
                page_number=metadata.get("page"),
                clause_number=metadata.get("clause_number") or None,
                section_title=metadata.get("section_title") or None,
                embedding=None,
            )
        )
    return records


def ingest_paths(
    settings: Settings,
    stores: KnowledgeStores,
    paths: Iterable[Path],
    *,
    source_type: str,
    tenant_id: str,
    collection_id: str | None = None,
) -> List[str]:
    object_store_backend = str(getattr(settings, "object_store_backend", "local")).lower()
    if object_store_backend != "local":
        raise NotImplementedError(
            f"OBJECT_STORE_BACKEND={object_store_backend!r} is not implemented for ingest yet."
        )

    ingested_doc_ids: List[str] = []
    effective_collection_id = collection_id or getattr(settings, "default_collection_id", "default")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue

        file_hash = _file_hash(path)
        title = path.name
        doc_id = make_doc_id(
            source_type=source_type,
            title=title,
            content_hash=file_hash,
            tenant_id=tenant_id,
        )
        if stores.doc_store.document_exists(doc_id, file_hash, tenant_id=tenant_id):
            continue

        raw_docs = _load_documents(path, settings)
        if not raw_docs:
            logger.info("No content extracted from %s; skipping.", path)
            continue

        for doc in raw_docs:
            doc.metadata = {
                **(doc.metadata or {}),
                "doc_id": doc_id,
                "title": title,
                "source_type": source_type,
                "source_path": str(path),
            }

        full_text = " ".join(doc.page_content for doc in raw_docs)
        structure = detect_structure(full_text)
        chunks = _split_with_structure(settings, raw_docs, structure)
        chunk_records = _build_chunk_records(chunks, doc_id, collection_id=effective_collection_id)
        stores.doc_store.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
                collection_id=effective_collection_id,
                title=title,
                source_type=source_type,
                content_hash=file_hash,
                source_path=str(path),
                num_chunks=len(chunk_records),
                ingested_at=dt.datetime.utcnow().isoformat() + "Z",
                file_type=path.suffix.lstrip(".").lower(),
                doc_structure_type=structure.doc_structure_type,
            )
        )
        try:
            stores.chunk_store.add_chunks(chunk_records, tenant_id=tenant_id)
        except Exception:
            stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
            raise
        ingested_doc_ids.append(doc_id)
    return ingested_doc_ids


def ensure_kb_indexed(settings: Settings, stores: KnowledgeStores, tenant_id: str) -> None:
    if not bool(getattr(settings, "seed_demo_kb_on_startup", True)):
        return
    default_collection_id = getattr(settings, "default_collection_id", "default")
    kb_docs = stores.doc_store.list_documents(
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=default_collection_id,
    )
    if kb_docs:
        return
    kb_dir = Path(getattr(settings, "kb_dir", Path("data") / "kb"))
    kb_paths = sorted(kb_dir.glob("*"))
    ingest_paths(
        settings,
        stores,
        kb_paths,
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=default_collection_id,
    )
