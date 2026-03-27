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


def _load_documents_docling(path: Path, settings: Settings) -> List[Document]:
    """Load a file using the Docling extraction engine.

    Docling handles PDF, DOCX, PPTX, XLSX, and images natively, producing
    layout-aware Markdown with preserved table structure.  Falls back to an
    empty list (triggering legacy loader) if Docling is not installed or if
    it fails to parse the file.

    Args:
        path:     Absolute path to the file to load.
        settings: Application settings; ``docling_ocr_enabled`` controls
                  whether Docling's internal OCR pipeline is active.

    Returns:
        A list of LangChain ``Document`` objects, one per Docling page/section,
        or an empty list on failure.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "Docling is not installed. Falling back to legacy loader for %s. "
            "Install with: pip install docling",
            path.name,
        )
        return []

    try:
        pdf_options = PdfPipelineOptions(do_ocr=settings.docling_ocr_enabled)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )
        result = converter.convert(str(path))
        if result is None or not hasattr(result, "document"):
            logger.warning("Docling returned no document for %s", path.name)
            return []

        # Export to Markdown — preserves table structure as GFM tables
        markdown_text = result.document.export_to_markdown()
        if not markdown_text.strip():
            return []

        return [Document(
            page_content=markdown_text,
            metadata={
                "source": str(path),
                "extraction_engine": "docling",
            },
        )]
    except Exception as exc:
        logger.warning("Docling failed to convert %s: %s. Falling back to legacy loader.", path.name, exc)
        return []


def _load_documents(path: Path, settings: Settings) -> List[Document]:
    """Load a file into LangChain Documents.

    When ``settings.extraction_engine == 'docling'``, Docling is tried first
    for supported formats (pdf, docx, pptx, xlsx).  If Docling is unavailable
    or fails, the call transparently falls through to the legacy pipeline.

    Legacy dispatch:
      .md / .txt       → TextLoader (unchanged)
      .pdf             → hybrid PyPDF + PaddleOCR when ocr_enabled, else PyPDFLoader
      .docx            → Docx2txtLoader (unchanged)
      image suffixes   → PaddleOCR when ocr_enabled, else skip (return [])
      other            → TextLoader with autodetect_encoding (fallback)
    """
    suffix = path.suffix.lower()

    # ── Docling fast-path ────────────────────────────────────────────────
    _DOCLING_SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}
    if getattr(settings, "extraction_engine", "legacy") == "docling" and suffix in _DOCLING_SUPPORTED:
        docs = _load_documents_docling(path, settings)
        if docs:
            return docs
        # Fall through to legacy pipeline on empty/failure

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
        chunk_records = _build_chunk_records(chunks, doc_id)

        # Persist parent doc row first (chunks.doc_id has FK -> documents.doc_id).
        stores.doc_store.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                tenant_id=tenant_id,
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
    if settings.object_store_backend != "local":
        raise NotImplementedError(
            f"OBJECT_STORE_BACKEND={settings.object_store_backend!r} is not implemented for KB indexing yet. "
            "Set OBJECT_STORE_BACKEND=local for now."
        )
    kb_docs = stores.doc_store.list_documents(source_type="kb", tenant_id=tenant_id)
    if kb_docs:
        return
    kb_paths = sorted(Path(settings.kb_dir).glob("*"))
    ingest_paths(settings, stores, kb_paths, source_type="kb", tenant_id=tenant_id)
