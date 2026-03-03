from __future__ import annotations

import json
from typing import Callable

from langchain.tools import tool

from agentic_chatbot.config import Settings
from agentic_chatbot.rag.stores import KnowledgeStores


def make_list_docs_tool(settings: Settings, stores: KnowledgeStores, session: object) -> Callable:
    @tool
    def list_indexed_docs(source_type: str = "") -> str:
        """List documents currently indexed in the knowledge base.

        Args:
          source_type: Optional filter — 'kb' for built-in knowledge base docs,
                       'upload' for user-uploaded docs. Leave empty for all.

        Returns:
          JSON list of docs with doc_id, title, source_type, num_chunks,
          file_type, doc_structure_type.
        """
        tenant_id = getattr(session, "tenant_id", settings.default_tenant_id)
        records = stores.doc_store.list_documents(source_type=source_type, tenant_id=tenant_id)
        docs = [
            {
                "doc_id": r.doc_id,
                "title": r.title,
                "source_type": r.source_type,
                "num_chunks": r.num_chunks,
                "file_type": r.file_type,
                "doc_structure_type": r.doc_structure_type,
            }
            for r in records
        ]
        return json.dumps(sorted(docs, key=lambda x: x["doc_id"]), ensure_ascii=False, indent=2)

    return list_indexed_docs
