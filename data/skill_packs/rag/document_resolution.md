# Document Resolution And Ambiguity Handling
agent_scope: rag
tool_tags: resolve_document, search_collection, fetch_document_outline
task_tags: document resolution, ambiguity, disambiguation
version: 1
enabled: true
description: Resolve fuzzy document references before deep retrieval.

## Workflow

Start with `resolve_document` whenever the user references a document by name, nickname, or description. If multiple candidates appear, narrow the scope with title words, source type, or collection ID before deeper search.

## Recovery

If the document reference stays ambiguous, explain the competing candidates and ask which one to use. Do not silently choose a weak match when the downstream answer depends on the exact source.
