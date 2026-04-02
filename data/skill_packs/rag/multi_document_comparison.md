# Multi-Document Comparison Workflows
agent_scope: rag
tool_tags: diff_documents, compare_clauses, search_collection
task_tags: comparison, diff, cross-document analysis
version: 1
enabled: true
description: Compare multiple sources without collapsing them too early.

## Comparison Pattern

Analyze each source independently first, then synthesize the comparison. Use `diff_documents` to establish structural overlap before drilling into clause-level differences with `compare_clauses`.

## Output

Call out what is shared, what differs, and what exists in only one source. Keep provenance attached to each finding so the final synthesizer can preserve citations cleanly.
