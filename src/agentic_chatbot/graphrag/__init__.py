"""Microsoft GraphRAG integration — knowledge graph indexing and search.

This package wraps the ``graphrag`` CLI to provide:
  - Automatic indexing at document ingest time (background thread)
  - Local search (entity-centric) and global search (community summaries)
  - Configuration generation from the application's Settings

GraphRAG runs as a parallel pipeline alongside pgvector — documents are
indexed into BOTH systems. The RAG agent gets ``graph_search_local`` and
``graph_search_global`` tools to query the knowledge graph.

Enable with: GRAPHRAG_ENABLED=true
"""
