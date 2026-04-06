from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.rag.clause_splitter import clause_split
from agentic_chatbot_next.rag.ocr import IMAGE_SUFFIXES, load_image_documents, load_pdf_documents_with_ocr
from agentic_chatbot_next.rag.structure_detector import REQUIREMENT_PATTERN, StructureAnalysis, detect_structure
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.rag.stores import KnowledgeStores, make_doc_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KBCoverageStatus:
    tenant_id: str
    collection_id: str
    configured_source_paths: tuple[str, ...]
    missing_source_paths: tuple[str, ...]
    indexed_source_paths: tuple[str, ...]
    indexed_doc_count: int
    sync_attempted: bool = False
    sync_error: str = ""
    synced_doc_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.sync_error and not self.missing_source_paths

    @property
    def status(self) -> str:
        return "ready" if self.ready else "not_ready"

    @property
    def reason(self) -> str:
        if self.sync_error:
            return "kb_sync_failed" if self.sync_attempted else "kb_status_check_failed"
        if self.missing_source_paths:
            return "kb_coverage_missing"
        return "ready"

    @property
    def suggested_fix(self) -> str:
        return f"python run.py sync-kb --collection-id {self.collection_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "tenant_id": self.tenant_id,
            "collection_id": self.collection_id,
            "configured_source_count": len(self.configured_source_paths),
            "indexed_doc_count": self.indexed_doc_count,
            "missing_sources": list(self.missing_source_paths),
            "sync_attempted": self.sync_attempted,
            "sync_error": self.sync_error,
            "suggested_fix": self.suggested_fix,
        }


def iter_kb_source_paths(settings: Settings) -> List[Path]:
    roots = [Path(settings.kb_dir), *(Path(path) for path in getattr(settings, "kb_extra_dirs", ()))]
    paths: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _normalized_source_paths(paths: Sequence[Path]) -> tuple[str, ...]:
    return tuple(sorted(str(path.resolve()) for path in paths))


def _record_title(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("title") or "")
    return str(getattr(record, "title", "") or "")


def _record_source_path(record: Any) -> str:
    if isinstance(record, dict):
        value = record.get("source_path")
    else:
        value = getattr(record, "source_path", "")
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return str(value)


def build_kb_coverage_status(
    settings: Settings,
    indexed_records: Sequence[Any],
    *,
    tenant_id: str,
    collection_id: str | None = None,
    sync_attempted: bool = False,
    sync_error: str = "",
    synced_doc_ids: Sequence[str] = (),
) -> KBCoverageStatus:
    effective_collection_id = collection_id or getattr(settings, "default_collection_id", "default")
    configured_paths = iter_kb_source_paths(settings)
    configured_source_paths = _normalized_source_paths(configured_paths)
    indexed_source_paths = tuple(
        sorted(
            {
                source_path
                for source_path in (_record_source_path(record) for record in indexed_records)
                if source_path
            }
        )
    )
    indexed_titles = {
        title
        for title in (_record_title(record) for record in indexed_records)
        if title
    }
    missing_source_paths = tuple(
        path
        for path in configured_source_paths
        if path not in indexed_source_paths and Path(path).name not in indexed_titles
    )
    return KBCoverageStatus(
        tenant_id=tenant_id,
        collection_id=effective_collection_id,
        configured_source_paths=configured_source_paths,
        missing_source_paths=missing_source_paths,
        indexed_source_paths=indexed_source_paths,
        indexed_doc_count=len(indexed_records),
        sync_attempted=sync_attempted,
        sync_error=str(sync_error or "").strip(),
        synced_doc_ids=tuple(str(doc_id) for doc_id in synced_doc_ids if str(doc_id)),
    )


def get_kb_coverage_status(
    settings: Settings,
    stores: KnowledgeStores,
    *,
    tenant_id: str,
    collection_id: str | None = None,
) -> KBCoverageStatus:
    effective_collection_id = collection_id or getattr(settings, "default_collection_id", "default")
    try:
        records = stores.doc_store.list_documents(
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_collection_id,
        )
    except Exception as exc:
        return build_kb_coverage_status(
            settings,
            [],
            tenant_id=tenant_id,
            collection_id=effective_collection_id,
            sync_error=str(exc),
        )
    return build_kb_coverage_status(
        settings,
        list(records),
        tenant_id=tenant_id,
        collection_id=effective_collection_id,
    )


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


def ensure_kb_indexed(
    settings: Settings,
    stores: KnowledgeStores,
    tenant_id: str,
    *,
    collection_id: str | None = None,
    attempt_sync: bool | None = None,
) -> KBCoverageStatus:
    effective_collection_id = collection_id or getattr(settings, "default_collection_id", "default")
    status = get_kb_coverage_status(
        settings,
        stores,
        tenant_id=tenant_id,
        collection_id=effective_collection_id,
    )
    if status.ready or not status.missing_source_paths:
        return status

    should_sync = (
        bool(getattr(settings, "seed_demo_kb_on_startup", True))
        if attempt_sync is None
        else bool(attempt_sync)
    )
    if not should_sync:
        return status

    missing_paths = [Path(path) for path in status.missing_source_paths]
    try:
        synced_doc_ids = ingest_paths(
            settings,
            stores,
            missing_paths,
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_collection_id,
        )
    except Exception as exc:
        return build_kb_coverage_status(
            settings,
            [],
            tenant_id=tenant_id,
            collection_id=effective_collection_id,
            sync_attempted=True,
            sync_error=str(exc),
        )

    refreshed = get_kb_coverage_status(
        settings,
        stores,
        tenant_id=tenant_id,
        collection_id=effective_collection_id,
    )
    return KBCoverageStatus(
        tenant_id=refreshed.tenant_id,
        collection_id=refreshed.collection_id,
        configured_source_paths=refreshed.configured_source_paths,
        missing_source_paths=refreshed.missing_source_paths,
        indexed_source_paths=refreshed.indexed_source_paths,
        indexed_doc_count=refreshed.indexed_doc_count,
        sync_attempted=True,
        sync_error=refreshed.sync_error,
        synced_doc_ids=tuple(str(doc_id) for doc_id in synced_doc_ids if str(doc_id)),
    )
