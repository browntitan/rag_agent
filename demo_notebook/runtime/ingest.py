from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import NotebookSettings
from .stores import ChunkRecord, DocumentRecord, PostgresVectorStore, make_doc_id, utcnow_iso


_CLAUSE_RE = re.compile(r"(?im)^(?:clause|section|article)\s+([0-9]+(?:\.[0-9]+)*)")
_SECTION_RE = re.compile(r"(?im)^([0-9]+(?:\.[0-9]+)*)\s+([A-Z][^\n]{2,100})$")


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1_048_576), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(settings: NotebookSettings, text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=False,
    )
    return splitter.split_text(text)


def _extract_clause(chunk: str) -> str | None:
    m = _CLAUSE_RE.search(chunk)
    if m:
        return m.group(1)
    m = _SECTION_RE.search(chunk)
    if m:
        return m.group(1)
    return None


def _extract_section_title(chunk: str) -> str | None:
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    if len(first) < 6:
        return None
    if first.lower().startswith(("clause", "section", "article")):
        return first
    if _SECTION_RE.match(first):
        return first
    return None


def index_kb_corpus(
    settings: NotebookSettings,
    store: PostgresVectorStore,
    *,
    reindex: bool = False,
) -> Dict[str, int]:
    kb_paths = sorted(p for p in Path(settings.kb_dir).glob("*") if p.is_file())

    existing = {doc.source_path: doc for doc in store.list_documents()}

    ingested = 0
    skipped = 0

    for path in kb_paths:
        if path.suffix.lower() not in {".md", ".txt"}:
            skipped += 1
            continue

        content_hash = _file_hash(path)
        path_str = str(path.resolve())
        existing_doc = existing.get(path_str)
        if existing_doc and existing_doc.content_hash == content_hash and not reindex:
            skipped += 1
            continue

        text = _load_text(path)
        chunks = _chunk_text(settings, text)
        if not chunks:
            skipped += 1
            continue

        doc_id = make_doc_id(path_str, content_hash)

        doc_record = DocumentRecord(
            doc_id=doc_id,
            title=path.name,
            source_path=path_str,
            source_type="kb",
            content_hash=content_hash,
            num_chunks=len(chunks),
            ingested_at=utcnow_iso(),
        )
        store.upsert_document(doc_record)

        embeddings = store.embeddings.embed_documents(chunks)
        chunk_records: List[ChunkRecord] = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_records.append(
                ChunkRecord(
                    chunk_id=f"{doc_id}#chunk{idx:04d}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=chunk,
                    embedding=emb,
                    clause_number=_extract_clause(chunk),
                    section_title=_extract_section_title(chunk),
                )
            )

        try:
            store.add_chunks(chunk_records)
        except Exception:
            store.delete_document(doc_id)
            raise

        ingested += 1

    return {
        "files_seen": len(kb_paths),
        "ingested": ingested,
        "skipped": skipped,
        "total_docs": len(store.list_documents()),
    }
