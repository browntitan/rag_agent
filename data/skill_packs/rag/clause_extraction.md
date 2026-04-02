# Clause And Requirement Extraction Workflows
agent_scope: rag
tool_tags: fetch_document_outline, extract_clauses, extract_requirements
task_tags: clause extraction, requirement extraction, structured docs
version: 1
enabled: true
description: Extract structured clauses and requirements reliably.

## Clause Flow

Use `fetch_document_outline` before `extract_clauses` when the exact clause number is unclear. That prevents you from requesting nonexistent clauses and gives you the right section titles for synthesis.

## Requirement Flow

Use `extract_requirements` when the user asks for all obligations, shall statements, or explicit requirements. If the document is not well structured, combine that output with targeted search to verify missing sections.
