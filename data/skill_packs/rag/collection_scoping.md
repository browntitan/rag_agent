# Metadata And Collection Scoping
agent_scope: rag
tool_tags: list_collections, search_collection, resolve_document
task_tags: metadata, collection scoping, namespace
version: 1
enabled: true
description: Scope retrieval to the right corpus before synthesizing.

## Scoping

Call `list_collections` when the corpus may contain multiple namespaces. If the user is working within one domain or project, scope retrieval to that collection early so irrelevant corpora do not pollute ranking.

## Guardrail

When a request could span multiple collections, say which collections you searched. If you only searched one collection because the user implied a scope, make that explicit in the answer.
