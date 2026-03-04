# C4 Architecture (Standalone demo_notebook)

This C4 set describes only the isolated notebook deliverable under `demo_notebook/`.

## C4 Level 1: System Context

```mermaid
flowchart LR
    U["Notebook User"]
    NB["demo_notebook System"]
    LLM["LLM Provider (Azure, Ollama, or vLLM)"]
    PG["PostgreSQL + pgvector"]
    KB["Local KB Files (data/kb)"]

    U -->|"Runs notebook cells"| NB
    NB -->|"Chat + tool calls"| LLM
    NB -->|"Read/write vectors"| PG
    NB -->|"Ingest corpus"| KB
```

## C4 Level 2: Container View

```mermaid
flowchart LR
    U["Notebook User"]

    subgraph SYS["demo_notebook System"]
      JN["Jupyter Notebook UI\n(agentic_rag_showcase.ipynb)"]
      RT["Runtime Package\n(demo_notebook/runtime)"]
    end

    LLM["Provider Endpoint\n(Azure or Ollama or vLLM)"]
    DB["PostgreSQL Container/Service\n(pgvector)"]
    FILES["KB Files\n(data/kb)"]

    U -->|"Execute scenarios"| JN
    JN -->|"Calls orchestrator"| RT
    RT -->|"Model + embeddings requests"| LLM
    RT -->|"Persist/query chunks"| DB
    RT -->|"Read markdown/txt docs"| FILES
```

## C4 Level 3: Component View (Runtime Package)

```mermaid
flowchart TD
    CFG["config.py\nsettings loader"]
    PROV["providers.py\nprovider factory"]
    ORCH["orchestrator.py\nDemoOrchestrator"]
    ROUTE["router.py\nBASIC vs AGENT"]
    GRAPH["graph_builder.py\nLangGraph assembly"]
    SUP["supervisor.py\nsupervisor node"]
    RAG["rag_agent.py\nRAG specialist"]
    GEN["general_agent.py\nGeneralAgent"]
    TOOLS["tools.py\ncalculator + RAG tools"]
    STORES["stores.py\nPostgresVectorStore"]
    INGEST["ingest.py\nchunk + index pipeline"]
    OBS["observability.py\nprint traces"]
    SK["skills.py\nprompt composition"]

    CFG --> PROV
    CFG --> ORCH
    CFG --> STORES
    CFG --> SK

    ORCH --> ROUTE
    ORCH --> GRAPH
    ORCH --> GEN
    ORCH --> INGEST
    ORCH --> OBS
    ORCH --> SK

    GRAPH --> SUP
    GRAPH --> RAG
    GRAPH --> TOOLS
    GEN --> TOOLS
    RAG --> TOOLS

    TOOLS --> STORES
    INGEST --> STORES
```

## Data Flow by Scenario

1. BASIC route
- Notebook calls `DemoOrchestrator.process_turn(...)`.
- `router.py` selects `BASIC`.
- Direct chat response from provider; no graph execution.

2. AGENT RAG route
- Router selects `AGENT`.
- Graph enters supervisor and routes to `rag_agent`.
- RAG tools perform retrieval from `dn_chunks`; response includes citations.

3. Parallel route
- Supervisor selects `parallel_rag`.
- Planner fans out to multiple RAG workers.
- Synthesizer merges worker outputs into one answer.

4. GeneralAgent direct route
- Notebook calls `run_general_agent_direct(...)`.
- GeneralAgent may invoke calculator/list-doc/rag-agent-tool chain.

## Skills note

`skills.py` is only applied when showcase mode is enabled. Baseline demos keep default prompt behavior.
