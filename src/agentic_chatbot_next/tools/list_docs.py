from __future__ import annotations

import json
from typing import Callable

from langchain.tools import tool


def _demo_group_for_title(title: str) -> str:
    lower = title.lower()
    if "runbook" in lower or "playbook" in lower:
        return "runbooks"
    if lower.startswith("api_") or "api" in lower:
        return "api_references"
    if any(token in lower for token in ("agreement", "contract", "addendum", "schedule", "msa", "dpa")):
        return "contracts"
    if any(token in lower for token in ("security", "privacy", "compliance", "control", "incident")):
        return "security_compliance"
    return "other"


def make_list_docs_tool(settings: object, stores: object, session: object) -> Callable:
    @tool
    def list_indexed_docs(source_type: str = "") -> str:
        """List documents currently indexed in the knowledge base."""

        tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
        records = stores.doc_store.list_documents(source_type=source_type, tenant_id=tenant_id)
        if getattr(session, "demo_mode", False):
            grouped = {
                "contracts": [],
                "security_compliance": [],
                "runbooks": [],
                "api_references": [],
                "other": [],
            }
            for record in records:
                grouped[_demo_group_for_title(record.title)].append(
                    {"doc_id": record.doc_id, "title": record.title}
                )
            for key in grouped:
                grouped[key] = sorted(grouped[key], key=lambda item: item["title"].lower())
            return json.dumps(
                {
                    "total_documents": len(records),
                    "source_type_filter": source_type or "all",
                    "groups": grouped,
                },
                ensure_ascii=False,
            )

        docs = [
            {
                "doc_id": record.doc_id,
                "title": record.title,
                "source_type": record.source_type,
                "num_chunks": record.num_chunks,
                "file_type": record.file_type,
                "doc_structure_type": record.doc_structure_type,
            }
            for record in records
        ]
        return json.dumps(sorted(docs, key=lambda item: item["doc_id"]), ensure_ascii=False, indent=2)

    return list_indexed_docs
