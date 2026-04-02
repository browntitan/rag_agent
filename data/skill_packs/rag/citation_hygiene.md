# Citation Hygiene And Synthesis
agent_scope: rag
tool_tags: fetch_chunk_window, search_collection, search_document
task_tags: citations, synthesis, grounded answers
version: 1
enabled: true
description: Keep final answers grounded in retrieved evidence.

## Citation Rules

Only cite evidence that directly supports the sentence you are writing. If the answer combines findings from multiple chunks, keep the claims attributable instead of blending them into unsupported prose.

## Synthesis Rules

Summarize first, then attach the evidence trail. If the retrieved evidence is partial or conflicting, say that clearly and limit the claim to what the sources support.
