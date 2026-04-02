# Retrieval Strategy Selection
agent_scope: rag
tool_tags: search_collection, search_document, search_all_documents
task_tags: retrieval strategy, vector, keyword, hybrid
version: 1
enabled: true
description: Pick the right retrieval strategy for the query shape.

## Strategy

Use `hybrid` by default when the user asks a broad semantic question. Use `keyword` when the user names exact phrases, identifiers, clause numbers, or field names. Use `vector` when the user describes a concept without exact wording.

## Iteration

If the first search misses obvious evidence, switch strategy instead of only repeating the same call. Broad search first, then narrow to a document or collection once relevant hits appear.
