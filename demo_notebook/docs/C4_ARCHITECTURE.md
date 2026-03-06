# C4 Architecture (Standalone demo_notebook)

This C4 set describes only the isolated notebook deliverable under `demo_notebook/`.

## C4 Level 1: System Context

```mermaid
flowchart LR
    U["Notebook User"]
    NB["demo_notebook System"]
    LLM["Model Provider APIs\n(Azure OpenAI, Ollama, or vLLM)"]
    PG["PostgreSQL + pgvector"]
    KB["Local KB Corpus\n(data/kb/*.md, *.txt)"]

    U -->|"Runs notebook scenarios"| NB
    NB -->|"Chat + embeddings calls"| LLM
    NB -->|"Vector + metadata persistence"| PG
    NB -->|"Reads KB files for indexing"| KB
```

## C4 Level 2: Container View

```mermaid
flowchart LR
    U["Notebook User"]

    subgraph SYS["demo_notebook System"]
      JN["Jupyter Notebook UI\n(agentic_rag_showcase.ipynb)"]
      RT["Runtime Package\n(demo_notebook/runtime)"]
      ENV["Notebook .env\n(provider, TLS, tiktoken, skills toggles)"]
    end

    subgraph DS["Optional Docker Support Stack\n(demo_notebook/docker-compose.yml)"]
      PGC["notebook-postgres\n(pgvector/pgvector:pg16)"]
      OLC["notebook-ollama (optional)\n(ollama/ollama)"]
    end

    EXTLLM["External Provider Endpoint\n(Azure or vLLM or Host Ollama)"]
    KB["KB Files\n(data/kb)"]

    U -->|"Execute cells"| JN
    JN -->|"Imports runtime + invokes orchestrator"| RT
    ENV -->|"Loaded by config.py"| RT
    RT -->|"Read/write dn_documents, dn_chunks"| PGC
    RT -->|"Model calls (if container Ollama path)"| OLC
    RT -->|"Model calls (if Azure/vLLM/host Ollama path)"| EXTLLM
    RT -->|"Ingest chunk source"| KB
```

## C4 Level 3: Component View (Runtime Package)

```mermaid
flowchart TD
    CFG["config.py\nsettings + env loading\nTLS vars + tiktoken cache env"]
    PROV["providers.py\nprovider factory\n(build_httpx_client,\nazure/ollama/vllm clients)"]
    ORCH["orchestrator.py\nDemoOrchestrator\n(bootstrap_kb, process_turn,\nBASIC/AGENT execution, fallback)"]
    SK["skills.py\nskill file loading + prompt composition\n(showcase-mode gated)"]
    OBS["observability.py\nprint callbacks\n[NOTEBOOK]/[ROUTER]/[GRAPH]"]

    ROUTE["router.py\ndeterministic route decision\n(BASIC vs AGENT)"]
    GRAPH["graph_builder.py\nLangGraph assembly\n(supervisor, rag, utility,\nparallel planner/worker/synthesizer)"]
    SUP["supervisor.py\nrouting JSON + heuristic fallback\n+ deterministic final_answer termination"]
    RAG["rag_agent.py\nRAG specialist ReAct agent"]
    GEN["general_agent.py\nGeneralAgent ReAct agent"]
    TOOLS["tools.py\ncalculator + retrieval tools\n(resolve/search/diff/compare/etc)"]

    STORE["stores.py\nPostgresVectorStore\n(dn_documents, dn_chunks)"]
    INGEST["ingest.py\nfile hash, chunking,\nclause extraction, embedding insert"]

    LLM["LLM + Embeddings Backends\nAzure / Ollama / vLLM"]
    KB["KB Files\n(data/kb)"]
    DB["PostgreSQL + pgvector"]

    CFG --> PROV
    CFG --> ORCH
    CFG --> SK
    CFG --> STORE

    SK --> ORCH
    ORCH --> OBS
    ORCH --> ROUTE
    ORCH --> GRAPH
    ORCH --> GEN
    ORCH --> INGEST

    GRAPH --> SUP
    GRAPH --> RAG
    GRAPH --> GEN
    GRAPH --> TOOLS

    RAG --> TOOLS
    GEN --> TOOLS
    TOOLS --> STORE
    INGEST --> STORE
    INGEST --> KB

    PROV --> LLM
    RAG --> LLM
    GEN --> LLM
    SUP --> LLM
    STORE --> DB
```

## Runtime Execution Diagram (BASIC vs AGENT)

```mermaid
sequenceDiagram
    participant User as Notebook User
    participant NB as Notebook Cell
    participant Orch as DemoOrchestrator
    participant Router as Router
    participant Graph as LangGraph Runtime
    participant Tools as Tool Layer
    participant DB as Postgres
    participant LLM as Provider LLM

    User->>NB: Run scenario cell
    NB->>Orch: process_turn(prompt, force_agent?)
    Orch->>Router: route_message(...)
    Router-->>Orch: BASIC or AGENT

    alt BASIC
      Orch->>LLM: direct chat invoke
      LLM-->>Orch: final text
      Orch-->>NB: DemoTurnResult(route=BASIC)
    else AGENT
      Orch->>Graph: invoke(state)
      Graph->>LLM: supervisor / agent reasoning
      Graph->>Tools: tool calls
      Tools->>DB: retrieval/query
      DB-->>Tools: chunks + metadata
      Tools-->>Graph: evidence payloads
      Graph->>LLM: synthesis / finalization
      LLM-->>Graph: final answer
      Graph-->>Orch: final_answer
      Orch-->>NB: DemoTurnResult(route=AGENT)
    end
```

## Notes

1. Skills are optional and only applied when `NOTEBOOK_SKILLS_ENABLED=true` and `NOTEBOOK_SKILLS_SHOWCASE_MODE=true`.
2. TLS and corporate cert behavior is configured in `.env` and applied through `config.py` + provider HTTP client wiring.
3. The notebook store is isolated from the main app schema (`dn_documents`, `dn_chunks`).
