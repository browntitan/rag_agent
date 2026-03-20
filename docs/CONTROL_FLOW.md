# Control Flow: How a Request Moves Through the System

This document traces a single user message from the HTTP request all the way to the
response. It covers every major function, every loop, every branch, and every handoff.
It is written for someone who knows Python and has ML training experience, but is new to
web frameworks and the LangChain/LangGraph library.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Concepts You Need First](#2-concepts-you-need-first)
3. [Layer 1 — The HTTP Entry Point](#3-layer-1--the-http-entry-point-apimainpy)
4. [Layer 2 — The Orchestrator](#4-layer-2--the-orchestrator-agentsorchestratorphy)
5. [Layer 3 — The Router](#5-layer-3--the-router)
6. [Layer 4 — The Multi-Agent Graph](#6-layer-4--the-multi-agent-graph)
7. [Layer 5 — The Supervisor Loop](#7-layer-5--the-supervisor-loop-graphsupervisorpy)
8. [Layer 6 — The Agent Loops (ReAct)](#8-layer-6--the-agent-loops-react)
9. [The RAG Agent in Detail](#9-the-rag-agent-in-detail-ragagentpy)
10. [How to Read This Code](#10-how-to-read-this-code)

---

## 1. The Big Picture

This is a multi-agent AI chatbot. When a user sends a message, the system:

1. Receives it over HTTP (like any web API)
2. Decides whether the message needs complex reasoning or can be answered directly
3. If complex: assembles a team of specialist AI agents and lets them collaborate
4. Returns the final answer

The three main layers, from outside to inside:

```
                        ┌─────────────────────────────────────────────┐
User Message ──HTTP──►  │  GATEWAY (FastAPI)                          │
                        │  api/main.py                                │
                        │  - Parses the HTTP request                  │
                        │  - Builds a ChatSession                     │
                        │  - Calls process_turn()                     │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │  BRAIN (Orchestrator)                       │
                        │  agents/orchestrator.py                     │
                        │  - Handles uploads                          │
                        │  - Runs the router (BASIC or AGENT?)        │
                        │  - Calls the correct path                   │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │  SPECIALISTS (Multi-Agent Graph)            │
                        │  graph/builder.py + graph/supervisor.py     │
                        │  rag_agent | utility_agent | data_analyst   │
                        │  parallel_rag                               │
                        └─────────────────────────────────────────────┘
```

Every request passes through all three layers when in AGENT mode. Simple questions
(greetings, arithmetic, general knowledge) short-circuit at the Brain layer.

---

## 2. Concepts You Need First

### What is a "session"?

In an ML training loop you have a single run with its own state (model weights,
optimizer state, loss history). A **session** is the same idea for a conversation.
It holds everything that persists across multiple user messages in the same chat thread:
- The conversation history (list of messages)
- Uploaded file IDs
- A working memory dictionary (scratchpad)
- A reference to the filesystem workspace

In this codebase the session is a plain Python dataclass called `ChatSession`
(`agents/session.py`, line 12). It is NOT stored in a database between requests — the
API client is responsible for replaying the full message history on every request
(the OpenAI chat format works this way).

### What is FastAPI?

FastAPI is a Python web framework. You define functions and decorate them with
`@app.post("/some/path")` — FastAPI handles the HTTP server, parses JSON bodies,
validates types with Pydantic, and calls your function. The function returns a dict
or Pydantic model which FastAPI serialises to JSON. It is analogous to Flask but with
automatic type validation.

`Depends(get_runtime_or_503)` is FastAPI's dependency injection: it calls
`get_runtime_or_503()` before your handler and passes the result in. If the runtime
failed to initialize, it raises an HTTP 503 error before your handler is even called.
This is a clean way to share expensive objects (LLM clients, database connections)
across request handlers.

### What is LangChain?

LangChain is a Python library that wraps LLM providers (OpenAI, Anthropic, etc.) with
a common interface. The core object you will see everywhere is a **chat model** — call
it with a list of messages and it returns an AI message. LangChain adds:
- Tool binding: `llm.bind_tools(tools)` tells the LLM about available functions
- Message types: `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`
- Callbacks: hooks that fire on every LLM call (used here for Langfuse tracing)

### What is LangGraph?

LangGraph is a library built on top of LangChain for building **stateful, cyclical
computation graphs**. Think of it as a directed graph where:
- Each **node** is a Python function that takes state and returns a partial state update
- Each **edge** is either fixed (always go to node B after node A) or conditional
  (look at the state and decide which node to go to next)
- The **state** is a typed dictionary that flows through the graph and is updated
  by each node

This is similar to how you might think of a PyTorch computation graph, except:
- Nodes are arbitrary Python functions (not tensor ops)
- The graph can loop (a node can route back to an earlier node)
- State is mutable and persists across loop iterations

The key LangGraph class used here is `StateGraph`. You call `.add_node()`, `.add_edge()`,
`.add_conditional_edges()`, then `.compile()` to get a runnable graph.

### What is ReAct?

ReAct (Reason + Act) is a prompting pattern for LLMs that use tools. The loop is:

```
1. Agent receives a task
2. Agent THINKS (generates a reasoning trace)
3. Agent ACTS (calls a tool)
4. Agent OBSERVES the tool result
5. Repeat from step 2 until the agent decides it has enough information
6. Agent produces final answer
```

In LangGraph, `create_react_agent(llm, tools=tools)` builds a small 2-node graph
that implements this loop automatically:
- Node 1 (agent): calls the LLM. If the LLM output contains tool calls, route to node 2.
  If not, return the final answer.
- Node 2 (tools): executes the tool calls, appends ToolMessage results, routes back to node 1.

The loop is controlled by `recursion_limit` in the graph config. The default would run
forever; setting `recursion_limit=N` raises an error after N node visits (each LLM call
+ each tool execution counts as one visit).

---

## 3. Layer 1 — The HTTP Entry Point (`api/main.py`)

File: `src/agentic_chatbot/api/main.py`

The system exposes an OpenAI-compatible API endpoint. The main endpoint is:

```python
# api/main.py, line 260
@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionsRequest,          # parsed from JSON body
    runtime: Runtime = Depends(get_runtime_or_503),  # dependency-injected
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
):
```

`ChatCompletionsRequest` is a Pydantic model that matches the OpenAI chat format:
```python
class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]   # [{"role": "user", "content": "..."}, ...]
    stream: bool = False
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = {}   # "force_agent": True skips the router
```

The handler does these six steps in order:

**Step 1 — Validate the model name.**
```python
if request.model != runtime.settings.gateway_model_id:
    raise HTTPException(status_code=400, detail=...)
```

**Step 2 — Build a `RequestContext`.**
```python
ctx = get_request_context(runtime, conversation_id=x_conversation_id, ...)
```
`RequestContext` (`context.py`, line 9) is a frozen dataclass holding
`tenant_id`, `user_id`, `conversation_id`, and `request_id`. The `session_id` property
is computed as `f"{tenant_id}:{user_id}:{conversation_id}"`. It is the stable key for
looking up this conversation's workspace on disk.

**Step 3 — Convert OpenAI messages to LangChain messages.**
```python
history, user_text = _to_langchain_history(request.messages)
```
`_to_langchain_history()` splits the message list: all messages except the last one
become `history` (LangChain BaseMessage objects), and the last user message becomes
`user_text` (a plain string). This is because the orchestrator handles the last
message separately.

**Step 4 — Create a `ChatSession`.**
```python
session = ChatSession.from_context(ctx, messages=history)
```
`ChatSession` (`agents/session.py`) is the main state object for the conversation.
It starts with `workspace=None`; the orchestrator will open the workspace on the
first call.

**Step 5 — Call the orchestrator.**
```python
answer = runtime.bot.process_turn(
    session,
    user_text=user_text,
    upload_paths=[],
    force_agent=bool(request.metadata.get("force_agent", False))
)
```
`runtime.bot` is a `ChatbotApp` instance created at server startup. `process_turn()`
is the main entry point — covered in the next section.

**Step 6 — Return the response.**
```python
if request.stream:
    return StreamingResponse(_stream_chat_chunks(model, answer), ...)
payload = _build_openai_completion_payload(model, answer, prompt_tokens)
return JSONResponse(payload)
```

---

## 4. Layer 2 — The Orchestrator (`agents/orchestrator.py`)

File: `src/agentic_chatbot/agents/orchestrator.py`

`process_turn()` is the top-level coordinator. It always runs the same four-phase
setup before making any routing decision:

```python
# orchestrator.py, line 256
def process_turn(
    self,
    session: ChatSession,
    *,
    user_text: str,
    upload_paths: Optional[List[Path]] = None,
    force_agent: bool = False,
) -> str:
```

### Phase 1 — Open the session workspace (line ~270)

```python
if session.workspace is None and self.ctx.settings.workspace_dir is not None:
    ws = SessionWorkspace.for_session(session.session_id, self.ctx.settings.workspace_dir)
    ws.open()
    session.workspace = ws
```

On the first turn, this creates `data/workspaces/<session_id>/` on disk. Every
subsequent call with the same `session_id` skips this (workspace is already open).
The workspace directory is later bind-mounted into Docker containers so the data
analyst agent can read and write files across turns.

### Phase 2 — Ensure the knowledge base is indexed (line ~280)

```python
ensure_kb_indexed(self.ctx.settings, self.ctx.stores, tenant_id=session.tenant_id)
```

A one-time check that the PostgreSQL pgvector index exists for this tenant. This
is idempotent — it does nothing if the index already exists.

### Phase 3 — Ingest any file uploads (line ~284)

```python
if upload_paths:
    self.ingest_and_summarize_uploads(session, upload_paths)
```

`ingest_and_summarize_uploads()` does three things:
1. Calls `ingest_paths()` to chunk the files and store them in PostgreSQL (this is
   what makes them searchable by the RAG agent)
2. Copies the raw files into the session workspace so the data analyst can read them
3. Calls `run_rag_agent()` directly (bypassing the supervisor) to generate a summary
   of the uploaded documents, which is appended to `session.messages` as an
   `AIMessage`

### Phase 4 — Route the request (line ~287)

```python
if self.ctx.settings.llm_router_enabled:
    decision = route_message_hybrid(user_text, ...)
else:
    decision = route_message(user_text, ...)
```

`decision` is a `RouterDecision` dataclass with a `route` field: either `"BASIC"`
or `"AGENT"`. See Section 5 for how routing works.

### After routing — Two paths

**BASIC path (line ~321):**
```python
if decision.route == "BASIC":
    text = run_basic_chat(
        self.ctx.providers.chat,
        messages=session.messages,
        user_text=user_text,
        system_prompt=self._basic_chat_system_prompt,
        callbacks=callbacks,
    )
    session.messages.append(AIMessage(content=text))
    return text
```
`run_basic_chat()` is a single `llm.invoke(messages)` call. No tools. No loop.
The LLM generates one response and returns. This is the fastest path.

**AGENT path (line ~334):**
```python
text = self._run_multi_agent_graph(session, user_text, callbacks, suggested_agent=...)
if text is None:
    text = self._run_general_agent_fallback(session, user_text, callbacks)
```
`_run_multi_agent_graph()` builds and invokes the full LangGraph multi-agent system.
If the graph is unavailable (e.g. LLM doesn't support tool calling), it falls back
to a simpler single-agent loop. See Section 6.

At the very end, `process_turn()` optionally clears the scratchpad:
```python
if self.ctx.settings.clear_scratchpad_per_turn:
    session.clear_scratchpad()
```
The scratchpad is a `dict` on `ChatSession` used by agents as within-turn working
memory. Clearing it prevents carry-over between unrelated turns.

---

## 5. Layer 3 — The Router

### Deterministic router (`router/router.py`)

File: `src/agentic_chatbot/router/router.py`

The router decides whether a message needs the full agent system (`"AGENT"`) or can
be answered by a single LLM call (`"BASIC"`). The deterministic router is pure Python —
no LLM, very fast.

```python
# router.py, line 34
@dataclass(frozen=True)
class RouterDecision:
    route: str          # "BASIC" or "AGENT"
    confidence: float   # 0.0 to 1.0
    reasons: list[str]
    suggested_agent: str = ""   # hint to the supervisor: which specialist to try first
    router_method: str = "deterministic"
```

The decision tree, in order of priority:

```
route_message(user_text, has_attachments, explicit_force_agent)
    │
    ├─ force_agent=True?                  → AGENT (confidence 1.0)
    │
    ├─ has_attachments=True?              → AGENT (confidence 1.0)
    │
    ├─ Regex match _DATA_ANALYSIS_HINTS?  → AGENT, suggested_agent="data_analyst" (0.90)
    │   (words like "excel", "csv", "statistics", "analyze data")
    │
    ├─ Regex match _TOOL_VERBS?           → reasons list grows
    │   (words like "search", "find", "calculate", "summarize")
    │
    ├─ Regex match _CITATION_HINTS?       → reasons list grows
    │   (words like "cite", "evidence", "source", "reference")
    │
    ├─ Regex match _HIGH_STAKES_HINTS?    → reasons list grows
    │   (words like "medical", "legal", "financial", "security")
    │
    ├─ Message length > 600 chars?        → reasons list grows
    │
    ├─ Any reasons accumulated?           → AGENT (0.75 if 1 reason, 0.9 if multiple)
    │
    └─ No reasons at all                  → BASIC (confidence 0.85)
```

### Hybrid LLM router (`router/llm_router.py`)

File: `src/agentic_chatbot/router/llm_router.py`

The hybrid router wraps the deterministic router and adds an LLM tiebreaker for
ambiguous cases. The `confidence` threshold is configurable (`LLM_ROUTER_CONFIDENCE_THRESHOLD`,
default 0.70).

```
route_message_hybrid(user_text, ...)
    │
    ├─ force_agent or has_attachments?
    │   └─ YES → return deterministic AGENT immediately (fast path, no LLM)
    │
    ├─ Run deterministic router → det (RouterDecision)
    │
    ├─ det.confidence >= threshold (0.70)?
    │   └─ YES → return det as-is (the regex was confident enough)
    │
    └─ det.confidence < threshold?
        └─ Call judge LLM with structured output:
           {
             "route": "BASIC" | "AGENT",
             "confidence": float,
             "reasoning": str,
             "suggested_agent": str
           }
           ├─ If LLM call succeeds → return LLM's decision
           └─ If LLM call fails   → fallback to det (deterministic result)
```

The `suggested_agent` field from the router is passed all the way into the graph as a
pre-seed for `next_agent` in `AgentState`. This lets the supervisor skip its first
routing decision when the router is already confident about which specialist to use.

---

## 6. Layer 4 — The Multi-Agent Graph

File: `src/agentic_chatbot/agents/orchestrator.py` (`_run_multi_agent_graph`)
File: `src/agentic_chatbot/graph/builder.py`

`_run_multi_agent_graph()` is the function that builds and invokes the LangGraph:

```python
# orchestrator.py, line ~379
registry = AgentRegistry(self.ctx.settings)   # knows which agents are available
graph = build_multi_agent_graph(
    chat_llm=..., judge_llm=..., settings=..., stores=...,
    session=session, callbacks=callbacks, registry=registry,
)
initial_state = build_initial_state(session, user_text, suggested_agent=suggested_agent)
result = graph.invoke(initial_state, config={"callbacks": callbacks, "recursion_limit": 50})
```

### What `build_multi_agent_graph()` does

`build_multi_agent_graph()` (`graph/builder.py`, line 33) constructs the graph in
three steps:

**Step 1 — Create a SessionProxy.**
```python
session_proxy = SessionProxy(
    session_id=session.session_id,
    tenant_id=session.tenant_id,
    scratchpad=dict(session.scratchpad),
    uploaded_doc_ids=list(session.uploaded_doc_ids),
    workspace=getattr(session, "workspace", None),
)
```
`SessionProxy` (`graph/session_proxy.py`) is a lightweight copy of the session data
that agent tool factories can close over. It carries the workspace reference so all
agents share the same host directory. It is NOT the same as `ChatSession` — it is a
simpler duck-typed object that exposes only the fields tools need.

**Step 2 — Create node functions (closures).**
```python
supervisor_fn   = make_supervisor_node(chat_llm, settings, ...)
rag_agent_fn    = make_rag_agent_node(settings, stores, chat_llm, judge_llm, session_proxy=session_proxy, ...)
utility_fn      = make_utility_agent_node(chat_llm, settings, stores, session_proxy, ...)
rag_worker_fn   = make_rag_worker_node(settings, stores, chat_llm, judge_llm, ...)
rag_synthesizer = make_rag_synthesizer_node(chat_llm, settings=settings, ...)
data_analyst_fn = make_data_analyst_node(chat_llm, settings, stores, session_proxy, ...)  # if Docker
```
Each `make_*_node()` function is a **factory** that returns a closure. The closure
captures its dependencies (LLM clients, tools, session proxy) in its scope and accepts
`(state: AgentState) -> dict` as its signature — this is the interface LangGraph nodes
require.

**Step 3 — Wire the graph.**
```python
graph = StateGraph(AgentState)

graph.add_node("supervisor",       supervisor_fn)
graph.add_node("rag_agent",        rag_agent_fn)
graph.add_node("utility_agent",    utility_fn)
graph.add_node("parallel_planner", parallel_planner_node)
graph.add_node("rag_worker",       rag_worker_fn)
graph.add_node("rag_synthesizer",  rag_synthesizer_fn)
graph.add_node("data_analyst",     data_analyst_fn)  # if Docker

graph.add_edge(START, "supervisor")                  # always start at supervisor
graph.add_conditional_edges("supervisor", route_from_supervisor, {...})
graph.add_edge("rag_agent",    "supervisor")         # loop back
graph.add_edge("utility_agent","supervisor")         # loop back
graph.add_edge("data_analyst", "supervisor")         # loop back
graph.add_conditional_edges("parallel_planner", fan_out_rag_workers, [...])
graph.add_edge("rag_worker",   "rag_synthesizer")
graph.add_edge("rag_synthesizer", "supervisor")      # loop back

compiled = graph.compile()
```

The resulting topology looks like this:

```
START
  │
  ▼
supervisor ────────────────────────────────────────────► END
  │                                                       ▲
  │ (reads state["next_agent"])                           │
  │                                                       │
  ├──► rag_agent ────────────────────────────────────────┤
  │         │                                             │
  │         └──────────── back to supervisor ────────────┤
  │                                                       │
  ├──► utility_agent ────────────────────────────────────┤
  │         │                                             │
  │         └──────────── back to supervisor ────────────┤
  │                                                       │
  ├──► data_analyst ────────────────────────────────────►┤
  │         │                                             │
  │         └──────────── back to supervisor ────────────┤
  │                                                       │
  └──► parallel_planner                                   │
            │                                             │
            ├──► rag_worker_1 ─┐                         │
            ├──► rag_worker_2 ─┤                         │
            ├──► rag_worker_3 ─┼──► rag_synthesizer ────►┤
            └──► rag_worker_N ─┘         │               │
                                         └── back to supervisor
```

### What `AgentState` is (`graph/state.py`)

`AgentState` is the shared notepad that flows through the graph. Every node receives
the full state and returns a partial update (a dict with only the fields it changed).
LangGraph merges these updates back into the main state automatically.

```python
# graph/state.py, line 44
class AgentState(MessagesState):
    # Inherited from MessagesState:
    # messages: Annotated[list[AnyMessage], add_messages]
    # add_messages is a reducer that APPENDS new messages (not replaces)

    tenant_id:         str = "local-dev"
    session_id:        str = ""
    uploaded_doc_ids:  List[str] = []
    demo_mode:         bool = False

    scratchpad:        Dict[str, str] = {}    # last-write-wins

    next_agent:        str = ""               # routing signal: which node runs next
    rag_sub_tasks:     List[Dict] = []        # tasks for parallel RAG workers
    rag_results:       Annotated[List[Dict], merge_rag_results] = []  # custom reducer
    final_answer:      str = ""
```

The `messages` field uses the `add_messages` reducer: when a node returns
`{"messages": [new_msg]}`, LangGraph appends `new_msg` to the existing list rather
than replacing the whole list. This is how conversation history accumulates.

`rag_results` uses a custom reducer `merge_rag_results()` that supports an explicit
clear operation — necessary because the synthesizer needs to reset results after
processing them.

### What `build_initial_state()` does (`graph/builder.py`, line 225)

```python
return {
    "messages": list(session.messages) + [HumanMessage(content=user_text)],
    "tenant_id": session.tenant_id,
    "session_id": session.session_id,
    "uploaded_doc_ids": list(session.uploaded_doc_ids),
    "scratchpad": dict(session.scratchpad),
    "next_agent": suggested_agent,   # pre-seed from router, or ""
    "rag_sub_tasks": [],
    "rag_results": [],
    "final_answer": "",
}
```

This is the starting state for the graph invocation. It snapshots the session at the
moment the graph starts. Changes to `AgentState.scratchpad` inside the graph are
synced back to `session.scratchpad` after the graph finishes.

---

## 7. Layer 5 — The Supervisor Loop (`graph/supervisor.py`)

File: `src/agentic_chatbot/graph/supervisor.py`

The supervisor is a special LangGraph node that runs at the start and is returned to
after every specialist agent finishes. Its job is to read the conversation history and
decide what to do next.

### The supervisor's prompt

The supervisor receives a system prompt that tells it:
- What agents are available and what each one does
- The conversation history so far
- Any previous RAG results (if parallel RAG was used)
- What JSON format to respond in

The expected output from the supervisor LLM is:
```json
{
  "next_agent": "rag_agent",       // or: utility_agent, parallel_rag, data_analyst, __end__
  "reasoning": "The user is asking about a document clause...",
  "direct_answer": ""              // only if next_agent == "__end__"
}
```

For parallel RAG, the format is extended:
```json
{
  "next_agent": "parallel_rag",
  "rag_sub_tasks": [
    {"query": "...", "preferred_doc_ids": ["doc1"], "worker_id": "rag_worker_0"},
    {"query": "...", "preferred_doc_ids": ["doc2"], "worker_id": "rag_worker_1"}
  ]
}
```

### The supervisor loop in code

```python
# graph/supervisor.py, line 98
def make_supervisor_node(chat_llm, settings, callbacks=None, max_loops=5, registry=None):
    loop_count = 0   # nonlocal counter — tracks how many times supervisor has run

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        nonlocal loop_count
        loop_count += 1

        # Safety valve: if we've looped too many times, stop
        if loop_count > max_loops:
            return {"next_agent": "__end__", "final_answer": last_ai_message}

        # Assemble messages for the LLM
        supervisor_msgs = [SystemMessage(content=system_prompt)]
        supervisor_msgs.extend(state.get("messages", []))

        # Inject parallel RAG results if present
        if state.get("rag_results"):
            supervisor_msgs.append(SystemMessage(content=format_rag_results(...)))

        # Ask the LLM what to do next
        resp = chat_llm.invoke(supervisor_msgs, config={"callbacks": callbacks})

        # Parse the JSON response with fallbacks
        parsed = _parse_supervisor_response(resp.content, valid_agents=...)

        # Build the state update
        updates = {"next_agent": parsed["next_agent"]}
        if parsed["next_agent"] == "__end__":
            updates["final_answer"] = parsed.get("direct_answer") or resp.content
        if parsed["next_agent"] == "parallel_rag":
            updates["rag_sub_tasks"] = parsed.get("rag_sub_tasks", [])

        return updates

    return supervisor_node
```

### The `route_from_supervisor` conditional edge

After the supervisor node updates `next_agent` in the state, LangGraph calls the
conditional edge function to decide which node to execute next:

```python
# graph/builder.py, line 138
def route_from_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "__end__")
    if next_agent == "rag_agent":        return "rag_agent"
    elif next_agent == "utility_agent":  return "utility_agent"
    elif next_agent == "parallel_rag":   return "parallel_planner"
    elif next_agent == "data_analyst":   return "data_analyst"
    else:                                return END
```

The selected node runs, returns its messages, and the graph follows the fixed edge
back to "supervisor". This is the outer loop.

### The full outer loop, step by step

```
graph.invoke(initial_state)
    │
    ▼
[supervisor] loop_count=1
    LLM decides: "rag_agent"
    Returns: {"next_agent": "rag_agent"}
    │
    ▼ conditional edge: route_from_supervisor → "rag_agent"
    │
    ▼
[rag_agent_node]
    Runs run_rag_agent() (inner ReAct loop, see Section 8)
    Returns: {"messages": [AIMessage(content=answer)]}
    │
    ▼ fixed edge: rag_agent → supervisor
    │
    ▼
[supervisor] loop_count=2
    LLM sees updated history including the RAG answer
    Decides: "__end__" (question has been answered)
    Returns: {"next_agent": "__end__", "final_answer": "..."}
    │
    ▼ conditional edge: END
    │
    ▼
graph.invoke() returns final state
```

### Loop limits

There are two independent loop limits:

| Limit | Where | Default | What it stops |
|---|---|---|---|
| `max_loops` | supervisor's closure | 5 | Supervisor outer loop |
| `recursion_limit` | `graph.invoke(config={"recursion_limit": 50})` | 50 | Total node visits in the graph |

The `recursion_limit` is a LangGraph safety net. Every time any node executes, LangGraph
increments a counter. If it exceeds `recursion_limit`, LangGraph raises
`GraphRecursionError`. This prevents infinite loops in case the supervisor keeps routing
to agents indefinitely.

---

## 8. Layer 6 — The Agent Loops (ReAct)

Each specialist agent (rag_agent, utility_agent, data_analyst) uses LangGraph's
`create_react_agent()` to implement a ReAct tool-calling loop. This is the **inner**
loop — it runs inside a single agent node invocation, before control returns to the
supervisor.

### What `create_react_agent` builds

```python
from langgraph.prebuilt import create_react_agent

agent_graph = create_react_agent(llm, tools=my_tools)
result = agent_graph.invoke({"messages": [SystemMessage(...), HumanMessage(...)]})
```

Under the hood, `create_react_agent` builds a 2-node StateGraph:

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    ▼                                 │ (if tool calls present)
              [agent node]                            │
              llm.invoke(messages)                    │
                    │                                 │
         ┌──────────┴──────────┐                     │
         │                     │                     │
   No tool calls          Tool calls                 │
   in response            in response                │
         │                     │                     │
         ▼                     ▼                     │
        END             [tools node]                  │
                        execute each tool             │
                        append ToolMessage(s)          │
                        ──────────────────────────────┘
```

The LLM's response is inspected for tool call requests. If present, those calls are
executed and the results appended as `ToolMessage` objects. The LLM is called again
with the updated message list. This continues until the LLM produces a response with
no tool calls.

### How the recursion limit works for inner agents

Each specialist node computes its own recursion limit before calling
`create_react_agent`:

```python
# utility_node.py
recursion_limit = (settings.max_agent_steps + settings.max_tool_calls + 1) * 2 + 1
```

The formula is: `2 × (steps + tool_calls + 1) + 1` because each "step" in the ReAct
loop costs 2 node visits (one for the agent node + one for the tools node). The `+1`
in the multiplication accounts for the final agent response (which has no tool calls
and costs 1 node visit).

### Utility agent loop (`graph/nodes/utility_node.py`)

The utility agent is given four tools:
- `calculator` — safe arithmetic evaluation
- `list_docs_tool` — lists documents available in the KB
- `scratchpad_write`, `scratchpad_read`, `scratchpad_list` — working memory
- `search_skills` — looks up guidance from the skills library

It uses `create_react_agent` and the loop is straightforward: the LLM calls tools
until it has enough to answer, then produces its final response.

### Data analyst agent loop (`graph/nodes/data_analyst_node.py`)

The data analyst has 11 tools including `execute_code`, which runs Python inside a
Docker container. The inner tool loop is the same ReAct pattern, but tool calls
can trigger Docker container creation — each `execute_code` call is a full container
lifecycle (create, run, collect output, destroy).

---

## 9. The RAG Agent in Detail (`rag/agent.py`)

File: `src/agentic_chatbot/rag/agent.py`

The RAG agent is the most complex. It is invoked differently from the utility and data
analyst agents: rather than just calling `create_react_agent`, it has additional
phases before and after the tool-calling loop.

```python
# rag/agent.py, line 29
def run_rag_agent(
    settings, stores, *,
    llm, judge_llm, query, conversation_context,
    preferred_doc_ids, must_include_uploads,
    top_k_vector, top_k_keyword, max_retries,
    session, callbacks=None,
) -> Dict[str, Any]:    # returns the "RAG contract" dict
```

### Phase 1 — Load system prompt and create tools

```python
system_prompt = load_rag_agent_skills(settings)   # reads data/skills/rag_agent.md

rag_tools = make_all_rag_tools(stores, session, settings=settings)  # 12 core tools
# Plus 5 extended tools if available (query_rewriter, chunk_expander, etc.)
```

The 12 core tools (`tools/rag_tools.py`) all close over `stores` and `session`. They
include: `search_document`, `search_all_documents`, `extract_clauses`,
`extract_requirements`, `compare_clauses`, `diff_documents`, `resolve_document`,
`list_document_structure`, `follow_up_search`, `scratchpad_write`, `scratchpad_read`,
`search_skills`.

### Phase 2 — Bind tools to the LLM

```python
try:
    llm_with_tools = llm.bind_tools(rag_tools)
except Exception:
    llm_with_tools = None   # this LLM doesn't support tool calling
```

`bind_tools()` is a LangChain method that serializes the tool schemas into the LLM's
function-calling format. For OpenAI-compatible models, this populates the `tools`
field in the API request. If the LLM doesn't support this, `llm_with_tools` is None
and the code falls back to a simpler single-pass retrieval.

### Phase 3 — Construct initial messages

```python
task_msg = (
    f"QUERY: {query}\n"
    f"PREFERRED_DOC_IDS: {preferred_doc_ids or '(search all)'}\n"
    "Use the available tools to retrieve evidence, then produce your final answer."
)
msgs = [
    SystemMessage(content=system_prompt),   # RAG agent's instruction set
    HumanMessage(content=task_msg),         # the user's query, formatted for the agent
]
```

### Phase 4 — Run the ReAct tool loop

```python
if llm_with_tools is not None:
    from langgraph.prebuilt import create_react_agent

    graph = create_react_agent(llm, tools=rag_tools)

    recursion_limit = (max_steps + max_tool_calls + 1) * 2 + 1

    result = graph.invoke(
        {"messages": msgs},
        config={"callbacks": callbacks, "recursion_limit": recursion_limit},
    )
    msgs = result["messages"]   # capture all messages including tool results
```

After this block, `msgs` contains the full trace:
`[SystemMessage, HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(tool_call), ToolMessage, ..., AIMessage(final)]`

### Phase 5 — Synthesis

After the tool loop, the agent is asked to synthesize its findings into a structured
JSON response:

```python
synthesis_prompt = render_template(load_rag_synthesis_prompt(settings), {"ORIGINAL_QUERY": query})
msgs.append(HumanMessage(content=synthesis_prompt))

synth_resp = llm.invoke(msgs, config={"callbacks": callbacks})
synth_text = synth_resp.content

# Parse the JSON
obj = extract_json(synth_text)
answer_bundle = {
    "answer": obj["answer"],
    "used_citation_ids": obj["used_citation_ids"],
    "followups": obj["followups"],
    "confidence_hint": obj["confidence_hint"],
}
```

The synthesis step is separate from the tool loop because it asks the LLM to
consolidate all retrieved information into a single, structured response. This two-step
approach (retrieve → synthesize) produces more reliable structured output than asking
the LLM to produce JSON inline during the tool loop.

### Phase 6 — Build citations and return the contract

```python
retrieved_docs = _extract_docs_from_messages(msgs)
citations = build_citations(retrieved_docs)

return {
    "answer": answer_bundle["answer"],
    "citations": citations,             # list of {citation_id, title, snippet, location}
    "used_citation_ids": [...],
    "confidence": ...,
    "followups": [...],
    "warnings": [...],
    "retrieval_summary": {"steps": N, "tool_calls_used": M, ...}
}
```

This dict is called the **RAG contract**. It is a fixed interface between `run_rag_agent()`
and its callers. When the RAG node returns to the supervisor, the contract is rendered
to a human-readable string by `render_rag_contract()` in `graph/nodes/rag_node.py` and
appended as an `AIMessage` to the graph state.

---

## 10. How to Read This Code

### Suggested reading order

Start with the five most important files, in this order:

| Step | File | What to look for |
|---|---|---|
| 1 | `agents/session.py` | The `ChatSession` dataclass — understand what the "session" object is before anything else |
| 2 | `graph/state.py` | The `AgentState` — understand the shared notepad that flows through the graph |
| 3 | `agents/orchestrator.py` | `process_turn()` — the master control function; trace every if/else branch |
| 4 | `graph/builder.py` | `build_multi_agent_graph()` — how the graph is assembled and `route_from_supervisor()` — the routing logic |
| 5 | `graph/supervisor.py` | `make_supervisor_node()` and `supervisor_node()` — the outer loop |

After those five, pick the specialist you are most interested in:
- RAG: `rag/agent.py` → `tools/rag_tools.py`
- Utility: `graph/nodes/utility_node.py` → `tools/calculator.py`
- Data analyst: `graph/nodes/data_analyst_node.py` → `tools/data_analyst_tools.py` → `sandbox/docker_executor.py`

### How to trace a single request by hand

1. Start at `api/main.py`, line 260. Read the `chat_completions` function.
2. Find the call to `runtime.bot.process_turn(session, ...)` and jump to
   `agents/orchestrator.py`, the `process_turn` method.
3. In `process_turn`, find the `route_message` or `route_message_hybrid` call.
   For a request like "summarize the contract", the deterministic router will match
   `_CITATION_HINTS` and return `AGENT`.
4. Follow the AGENT path to `_run_multi_agent_graph()`.
5. Inside that method, find `graph.invoke(initial_state, ...)`. This is where the graph
   starts running.
6. The graph always enters at the `supervisor` node first. Go to `graph/supervisor.py`
   and read `supervisor_node()`. The LLM will produce `{"next_agent": "rag_agent"}`.
7. The conditional edge routes to `rag_agent`. Go to `graph/nodes/rag_node.py`,
   `rag_agent_node()`, which calls `run_rag_agent()` in `rag/agent.py`.
8. Inside `run_rag_agent`, find the `create_react_agent` call. This is the inner
   tool-calling loop.
9. After the loop ends, find the synthesis step and the `return` of the RAG contract.
10. Back in `rag_agent_node`, find `render_rag_contract(contract)`. The rendered text
    is returned as `{"messages": [AIMessage(content=rendered)]}`.
11. The fixed edge routes back to `supervisor`. The supervisor sees the RAG answer in
    the message history and returns `{"next_agent": "__end__", "final_answer": ...}`.
12. `graph.invoke()` returns. Back in `_run_multi_agent_graph`, `result["final_answer"]`
    is extracted and returned through `process_turn()` all the way to the API handler.

### The two questions to ask at every step

When reading any function in this codebase, ask:

1. **What goes in?**
   - What parameters does the function receive?
   - What fields of AgentState / ChatSession does it read?

2. **What comes out?**
   - What does the function return?
   - Which fields of AgentState / ChatSession does it modify?

These two questions map every function to its role in the control flow.

### Where the agent loop is

There are two loops at different levels:

**Outer loop (supervisor):**
```
Location: graph/supervisor.py — supervisor_node() is the loop body
          graph/builder.py — the edges create the cycle
Iteration: each time the supervisor runs and returns next_agent != "__end__"
Exit: next_agent == "__end__" OR loop_count > max_loops
```

**Inner loop (ReAct tool calling):**
```
Location: create_react_agent() inside langgraph.prebuilt
          Used by: rag/agent.py, graph/nodes/utility_node.py, graph/nodes/data_analyst_node.py
Iteration: each LLM call that produces a tool call
Exit: LLM produces response with no tool calls OR recursion_limit reached
```

In ML terms: the outer loop is like a training epoch (supervisor decides strategy),
the inner loop is like a forward pass with gradient updates (agent iterates using
tools until it converges on an answer).

---

## Quick Reference: Control Flow Pathways

### Shortest path (BASIC)
```
HTTP request
  → chat_completions()           [api/main.py:260]
  → process_turn()               [orchestrator.py:256]
  → route_message()              [router.py:44]          returns "BASIC"
  → run_basic_chat()             [basic_chat.py]         single llm.invoke()
  ← return text
```

### Standard path (AGENT, single RAG turn)
```
HTTP request
  → chat_completions()           [api/main.py:260]
  → process_turn()               [orchestrator.py:256]
  → route_message_hybrid()       [llm_router.py:117]     returns "AGENT"
  → _run_multi_agent_graph()     [orchestrator.py:361]
    → build_multi_agent_graph()  [builder.py:33]         compile graph
    → build_initial_state()      [builder.py:225]
    → graph.invoke()
      → supervisor_node()        [supervisor.py:115]     loop_count=1, next_agent="rag_agent"
      → rag_agent_node()         [rag_node.py:94]
        → run_rag_agent()        [rag/agent.py:29]
          → create_react_agent() inner ReAct loop
          → synthesis step
          ← RAG contract dict
        ← {"messages": [AIMessage]}
      → supervisor_node()        [supervisor.py:115]     loop_count=2, next_agent="__end__"
      ← {"next_agent": "__end__", "final_answer": "..."}
    ← result state
  ← answer text
← HTTP 200 response
```

### Parallel RAG path
```
  → supervisor_node()            returns next_agent="parallel_rag", rag_sub_tasks=[...]
  → parallel_planner_node()      splits tasks
  → fan_out_rag_workers()        creates Send(rag_worker, state) for each task
  → rag_worker_0/1/N             run simultaneously (LangGraph parallelism)
  → rag_synthesizer_node()       merges all results
  → supervisor_node()            loop_count+1
```
