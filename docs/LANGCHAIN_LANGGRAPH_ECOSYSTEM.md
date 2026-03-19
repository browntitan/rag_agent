# LangChain / LangGraph Ecosystem

Technical documentation covering the LangChain and LangGraph platform, how the key classes work, and how this codebase uses them to build a production multi-agent RAG system.

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [LangChain Core — Message System](#2-langchain-core--message-system)
3. [LangChain Core — LLM Interface](#3-langchain-core--llm-interface)
4. [LangChain Core — Tools](#4-langchain-core--tools)
5. [LangGraph — StateGraph](#5-langgraph--stategraph)
6. [LangGraph — Reducers and State Management](#6-langgraph--reducers-and-state-management)
7. [LangGraph — `create_react_agent`](#7-langgraph--create_react_agent)
8. [LangGraph — Send API (Parallel Fan-Out)](#8-langgraph--send-api-parallel-fan-out)
9. [Design Patterns in This Codebase](#9-design-patterns-in-this-codebase)
10. [Class Reference Table](#10-class-reference-table)

---

## 1. Platform Overview

### Package Relationships

```
langchain-core          # message types, base LLM interface, tool protocol
    ↓
langchain               # high-level chains, document loaders, retrievers
    ↓
langchain-community     # third-party integrations (pgvector, etc.)
    ↓
langchain-openai        # AzureChatOpenAI, OpenAIEmbeddings
langchain-ollama        # ChatOllama

langgraph               # stateful graph execution engine
    uses → langchain-core message types
    exposes → StateGraph, create_react_agent, Send
```

### What Each Package Provides in This Codebase

| Package | What it provides here |
|---|---|
| `langchain-core` | `BaseMessage` hierarchy, `BaseChatModel`, `@tool` decorator, `ToolMessage` |
| `langchain-openai` | `AzureChatOpenAI` for Azure OpenAI provider |
| `langchain-ollama` | `ChatOllama` for local Ollama LLMs |
| `langchain-community` | `PGVector` for PostgreSQL vector store |
| `langgraph` | `StateGraph`, `create_react_agent`, `Send`, `START`, `END`, `MessagesState` |
| `langgraph.prebuilt` | `create_react_agent` (prebuilt ReAct graph) |

---

## 2. LangChain Core — Message System

**Module:** `langchain_core.messages`

### Message Type Hierarchy

```
BaseMessage
├── HumanMessage      (.type = "human")
├── AIMessage         (.type = "ai")
│   └── .tool_calls   ← list of tool call requests from the LLM
├── SystemMessage     (.type = "system")
└── ToolMessage       (.type = "tool")
    └── .tool_call_id ← links back to the AIMessage.tool_calls entry
```

### Key Attributes

```python
# All messages
msg.type: str             # "human" | "ai" | "system" | "tool"
msg.content: str | list   # string or multimodal content list
msg.id: str               # unique message ID (used by add_messages reducer for dedup)

# AIMessage only
msg.tool_calls: List[{
    "id": str,            # tool call ID, must match ToolMessage.tool_call_id
    "name": str,          # tool function name
    "args": dict,         # JSON-deserialised arguments
}]
msg.additional_kwargs     # provider-specific metadata

# ToolMessage only
msg.tool_call_id: str     # matches the AIMessage.tool_calls[i]["id"]
msg.name: str             # tool name (informational)
```

### How Messages Flow Through the System

```
ChatSession.messages: List[BaseMessage]
  └── persists across turns (human/ai pairs accumulate)

AgentState.messages: Annotated[List[AnyMessage], add_messages]
  └── within-graph state; nodes append, never overwrite

create_react_agent input:
  {"messages": [SystemMessage(prompt), HumanMessage(query), ...history...]}
  └── the agent loop appends AIMessages and ToolMessages as it runs
```

### `add_messages` Reducer

Built into LangGraph. Behaviour:
- New messages with IDs not in current list → appended
- New messages with IDs matching existing messages → replace existing (update semantics)
- Preserves ordering

This makes it safe for parallel nodes to append messages without overwriting each other — each worker's `AIMessage` gets a unique ID.

### Usage in This Codebase

```python
# orchestrator.py — build conversation history
session.messages.append(AIMessage(content=rendered))

# supervisor.py — prepend system prompt each loop
supervisor_msgs = [SystemMessage(content=system_prompt)] + state["messages"]

# utility_node.py — inject system prompt without mutating state
msgs = [SystemMessage(content=system_prompt)] + state["messages"]
result = utility_graph.invoke({"messages": msgs})
```

---

## 3. LangChain Core — LLM Interface

**Module:** `langchain_core.language_models`

### `BaseChatModel`

The abstract base class for all chat LLMs. Key methods used in this codebase:

```python
# Synchronous invocation — returns AIMessage
response: AIMessage = llm.invoke(
    messages: List[BaseMessage],
    config: dict = {
        "callbacks": List[BaseCallbackHandler],
        "tags": List[str],           # added to Langfuse trace
        "metadata": dict,            # added to Langfuse trace
        "recursion_limit": int,      # passed to LangGraph graphs
    }
)

# Bind tools — returns a new LLM that includes tool schemas in the API call
llm_with_tools = llm.bind_tools(tools: List[BaseTool])
# This raises if the LLM wrapper doesn't support tool calling.
# Used in _is_graph_capability_error() detection.

# Structured output — LLM forced to return a specific schema
structured_llm = llm.with_structured_output(schema: Type[BaseModel] | dict)
result = structured_llm.invoke(messages)
# Used in llm_router.py for LLMRouterOutput parsing
```

### Provider Abstraction

**Module:** `src/agentic_chatbot/providers/llm_factory.py`

```python
@dataclass
class ProviderBundle:
    chat: BaseChatModel      # primary chat LLM (routing, agents, synthesis)
    judge: BaseChatModel     # judge LLM (grading, routing decisions)
    embeddings: Embeddings   # embedding model for vector search
```

`build_providers(settings)` selects the backend based on `settings.llm_provider`:
- `"azure"` → `AzureChatOpenAI(deployment_name=..., api_key=..., api_version=...)`
- `"ollama"` → `ChatOllama(model=settings.ollama_model, base_url=...)`

All agents receive `ProviderBundle.chat` for generation and `ProviderBundle.judge` for grading/routing. The abstraction means the agent code doesn't know which backend is running.

### Callbacks and Observability

`get_langchain_callbacks(settings, session_id, trace_name, metadata)` returns:
- Empty list if Langfuse keys are not configured
- `[LangfuseCallbackHandler(session_id, trace_name, metadata)]` if configured

The `callbacks` list is passed in the `config` dict to every LLM `.invoke()` call and to every LangGraph `.invoke()` call. LangGraph propagates callbacks through the entire graph automatically.

---

## 4. LangChain Core — Tools

**Module:** `langchain_core.tools`

### `@tool` Decorator

The primary way tools are defined in this codebase:

```python
from langchain_core.tools import tool

@tool
def search_chunks(query: str, top_k: int = 8) -> str:
    """Search the knowledge base for relevant chunks.

    Args:
        query: The search query.
        top_k: Maximum number of chunks to return.

    Returns:
        JSON string with matching chunks.
    """
    # implementation...
```

The `@tool` decorator:
1. Reads the function's `__doc__` string as the tool description (passed to the LLM as a capability description)
2. Reads type hints to build a JSON Schema `inputSchema` (what args the LLM must supply)
3. Reads the `Args:` section from the docstring to generate per-argument descriptions
4. Returns a `StructuredTool` instance with `.name`, `.description`, `.args_schema`, `.invoke()`

**Naming:** The function name becomes the tool name (e.g., `search_chunks`). The LLM references it by this name when generating `tool_calls`.

### Factory Pattern for Tool Closures

Most tools in this codebase are defined inside factory functions to close over `stores` and `session`:

```python
def make_all_rag_tools(settings, stores, session, llm, judge_llm) -> List[BaseTool]:

    @tool
    def search_chunks(query: str, top_k: int = 8) -> str:
        """..."""
        results = stores.chunk_store.similarity_search(query, top_k=top_k)
        # ...

    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """..."""
        session.scratchpad[key] = value
        # ...

    return [search_chunks, scratchpad_write, ...]
```

This means:
- `stores` (DB connections) are shared across tools in a single call
- `session.scratchpad` is isolated per agent invocation (each node creates its own `SessionProxy`)
- The tool functions themselves are stateless — all state lives in the closed-over objects

### Tool Execution in `create_react_agent`

When `create_react_agent` is invoked:

```
1. LLM receives: [SystemMessage, HumanMessage] + tool schemas in API payload
2. LLM returns: AIMessage with .tool_calls = [{"id": "call_abc", "name": "search_chunks", "args": {"query": "..."}}]
3. LangGraph routes to tools node
4. tools node calls: tool_map["search_chunks"].invoke({"query": "..."})
5. tools node produces: ToolMessage(content=result, tool_call_id="call_abc")
6. ToolMessage appended to messages
7. Messages sent back to LLM (next cycle)
8. LLM sees tool result, decides next action
9. Loop continues until LLM returns AIMessage with no tool_calls
```

### `ToolMessage`

```python
ToolMessage(
    content: str,           # tool return value (must be string)
    tool_call_id: str,      # matches AIMessage.tool_calls[i]["id"]
    name: str,              # tool name (informational)
)
```

Tool return values **must be strings**. This is why all tools serialize their output to JSON strings before returning, even if the underlying data is a dict or list:

```python
return json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
```

---

## 5. LangGraph — StateGraph

**Module:** `langgraph.graph`

### Core Concepts

`StateGraph` is a directed acyclic graph (with cycles supported via conditional edges) where:
- **Nodes** are Python functions: `(state) -> partial_state_update_dict`
- **Edges** define control flow between nodes
- **State** is a typed dict that flows through every node; nodes return partial updates that are merged

### `StateGraph(state_schema)`

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)   # AgentState is a TypedDict/dataclass
```

The `state_schema` class defines:
- All fields that exist in the state
- Reducers (via `Annotated`) for fields with non-default merge semantics
- Default values for fields not set by a node

### Node Functions

```python
def my_node(state: AgentState) -> Dict[str, Any]:
    # Read from state
    messages = state["messages"]

    # Do work...
    result = "..."

    # Return ONLY the fields you want to update
    return {
        "messages": [AIMessage(content=result)],  # merged by add_messages reducer
        "next_agent": "rag_agent",                # overwrites previous value
    }
```

**Critical rule:** Nodes return a *partial* update dict, not the full state. LangGraph merges the returned dict into the current state using each field's reducer. Fields not in the returned dict are unchanged.

### Adding Nodes and Edges

```python
# Nodes
graph.add_node("supervisor", supervisor_fn)
graph.add_node("rag_agent", rag_agent_fn)

# Unconditional edge: always goes from rag_agent back to supervisor
graph.add_edge("rag_agent", "supervisor")

# Entry point: START is a built-in sentinel
graph.add_edge(START, "supervisor")

# Conditional edge: supervisor decides where to go next
graph.add_conditional_edges(
    "supervisor",                    # source node
    route_from_supervisor,           # routing function: (state) -> str
    {
        "rag_agent": "rag_agent",    # routing fn return value → target node name
        "utility_agent": "utility_agent",
        END: END,                    # END is a built-in sentinel for termination
    }
)
```

### `compile()`

```python
compiled = graph.compile()
```

Compiles the graph into an executable object with:
- `.invoke(state, config)` — synchronous execution
- `.ainvoke(state, config)` — async execution
- `.stream(state, config)` — stream events
- `.get_graph()` — inspect topology

This codebase uses `.invoke()` exclusively (no async).

### Graph Topology in This Codebase

```
START
  ↓
supervisor ──→ rag_agent ──────────────→ supervisor
         ├──→ utility_agent ────────────→ supervisor
         ├──→ data_analyst ─────────────→ supervisor
         ├──→ parallel_planner
         │         ↓ (Send API fan-out)
         │    [rag_worker × N] ──────────→ rag_synthesizer ──→ supervisor
         └──→ END
```

---

## 6. LangGraph — Reducers and State Management

### What Is a Reducer?

A reducer is a function `(current_value, new_value) -> merged_value` used when LangGraph merges a node's returned partial state into the full state.

```python
from typing import Annotated

class AgentState(MessagesState):
    # Custom reducer for rag_results
    rag_results: Annotated[List[Dict], merge_rag_results] = []
    #             ^^^^^^^^              ^^^^^^^^^^^^^^^^
    #             type annotation       the reducer function
```

Without `Annotated`, fields use last-write-wins semantics: the node's returned value replaces the current value entirely.

### Built-in Reducer: `add_messages`

```python
from langgraph.graph.message import add_messages

messages: Annotated[List[AnyMessage], add_messages]
```

`add_messages` behaviour:
- New messages without matching IDs in current list → appended in order
- New messages whose IDs match existing messages → in-place update (idempotent)
- Keeps ordering stable even when parallel nodes both append messages

This is inherited by `AgentState` via `MessagesState`:
```python
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
```

### Custom Reducer: `merge_rag_results`

```python
def merge_rag_results(
    current: List[Dict] | None,
    new: List[Dict] | None,
) -> List[Dict]:
    has_clear = any(isinstance(item, dict) and item.get("__clear__") for item in (new or []))
    if has_clear:
        # Synthesizer is clearing the results after merging
        return [item for item in (new or []) if not item.get("__clear__")]
    # Parallel workers append their results
    return list(current or []) + list(new or [])
```

This supports two distinct operations:
1. **Parallel append:** Each worker calls `return {"rag_results": [one_result]}`. The reducer concatenates across all workers.
2. **Explicit clear:** The synthesizer calls `return {"rag_results": [{"__clear__": True}]}`. The reducer discards all accumulated results.

### Why Reducers Matter for Parallel Execution

Without reducers, parallel nodes would overwrite each other:
```
Worker 1 returns: {"rag_results": [result_1]}
Worker 2 returns: {"rag_results": [result_2]}
```

With last-write-wins, only one result would survive. With `merge_rag_results`, both are collected:
```
After merge: {"rag_results": [result_1, result_2]}
```

### `MessagesState` vs Custom `AgentState`

```python
# MessagesState is the minimal state base class in LangGraph
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# AgentState extends it with additional fields for this application
class AgentState(MessagesState):
    tenant_id: str
    session_id: str
    uploaded_doc_ids: List[str]
    demo_mode: bool
    scratchpad: Dict[str, str]      # last-write-wins
    next_agent: str                  # last-write-wins
    rag_sub_tasks: List[Dict]        # last-write-wins
    rag_results: Annotated[List[Dict], merge_rag_results]  # custom reducer
    final_answer: str                # last-write-wins
```

Fields without `Annotated` reducers use last-write-wins. Since only one node writes `next_agent` at a time (the supervisor), last-write-wins is correct and safe.

---

## 7. LangGraph — `create_react_agent`

**Module:** `langgraph.prebuilt`

`create_react_agent` builds a mini StateGraph internally. This is a prebuilt pattern for the ReAct (Reason + Act + Observe) agent loop.

### Internal Graph Topology

```
START
  ↓
agent_node   ← LLM call; produces AIMessage (possibly with tool_calls)
  ↓ (conditional)
  ├─ has tool_calls → tools_node → agent_node (loop)
  └─ no tool_calls  → END
```

### How It Works Step by Step

```python
agent_graph = create_react_agent(llm, tools=tool_list)

result = agent_graph.invoke(
    {"messages": [SystemMessage(system_prompt), HumanMessage(query)]},
    config={"recursion_limit": 25, "callbacks": callbacks},
)
# result["messages"] contains the full message history including tool calls and results
```

**Per cycle:**

1. `agent_node`: calls `llm.bind_tools(tools).invoke(messages)`
   - If response has `.tool_calls`: return partial update `{"messages": [AIMessage(tool_calls=[...])]}`
   - If response has no `.tool_calls`: return `{"messages": [AIMessage(content=answer)]}` → routes to END

2. `tools_node`: for each tool call in the last AIMessage:
   - Looks up `tool_call["name"]` in the tool map
   - Calls `tool.invoke(tool_call["args"])`
   - Appends `ToolMessage(content=str(result), tool_call_id=tool_call["id"])`

3. Loop back to `agent_node` with expanded messages list

### Recursion Limit Formula

```python
recursion_limit = (max(max_steps, max_tool_calls) + 1) * 2 + 1
```

Why: each ReAct cycle visits 2 nodes (agent + tools). The `+1` accounts for the final response-only agent call. LangGraph counts node visits, not cycles.

Example: `max_steps=10, max_tool_calls=12` → `(12+1)*2+1 = 27`

### System Prompt Injection

`create_react_agent` does not have a `system_prompt` parameter (in the versions used here). System prompts are injected by prepending a `SystemMessage` to the messages list before calling `.invoke()`:

```python
msgs = [SystemMessage(content=system_prompt)] + state["messages"]
result = utility_graph.invoke({"messages": msgs})
```

### Why Nodes Return `{"messages": [...]}`

`create_react_agent` uses `MessagesState` internally. Node functions return partial update dicts. The `messages` field uses `add_messages` reducer. So returning `{"messages": [AIMessage(...)]}` appends the new message to the existing list — it does not replace the list.

### How This Codebase Uses `create_react_agent`

| Location | Agent | Tools |
|---|---|---|
| `agents/general_agent.py` | `GeneralAgent` (fallback) | calculator, rag_agent_tool, list_docs, memory_* |
| `graph/nodes/utility_node.py` | Utility agent subgraph | calculator, list_docs, memory_* |
| `graph/nodes/data_analyst_node.py` | Data analyst subgraph | load_dataset, inspect_columns, execute_code, calculator, scratchpad_* |
| `rag/agent.py` | RAG agent | 11 core RAG tools + 5 extended tools |

In all cases, the `create_react_agent` subgraph is embedded as a node function inside the outer `StateGraph` (or as the entire agent for the fallback path).

---

## 8. LangGraph — Send API (Parallel Fan-Out)

**Module:** `langgraph.types`

### `Send` Object

```python
from langgraph.types import Send

Send(node: str, state: dict)
```

`Send` is a deferred node invocation instruction. It tells LangGraph: "create a new execution of `node` with the given `state` dict as its input."

### How Fan-Out Works

When a conditional edge routing function returns a list (instead of a string), LangGraph:
1. Checks if any items are `Send` objects
2. Schedules all `Send` targets for parallel execution
3. Waits for all parallel executions to finish
4. Merges their state updates using reducers

```python
# In builder.py
graph.add_conditional_edges(
    "parallel_planner",
    fan_out_rag_workers,
    ["rag_worker", "rag_synthesizer"],  # ← list form enables Send returns
)

def fan_out_rag_workers(state: AgentState) -> list:
    tasks = state["rag_sub_tasks"][:max_workers]
    sends = []
    for task in tasks:
        worker_state = dict(state)
        worker_state["rag_sub_tasks"] = [task]   # one task per worker
        worker_state["rag_results"] = []         # clean slate
        sends.append(Send("rag_worker", worker_state))
    return sends
```

### State Isolation in Workers

Each `Send` call receives a **copy** of the state. Workers cannot read each other's in-progress state. The only coordination happens through the shared `rag_results` reducer after all workers finish.

```
parallel_planner emits 3 Send("rag_worker", state_copy_i)
  ↓                ↓                ↓
  worker 1        worker 2          worker 3   (all run concurrently)
  │                │                 │
  returns          returns           returns
  {"rag_results": [r1]}  {"rag_results": [r2]}  {"rag_results": [r3]}
  │                │                 │
  └────────────────┴─────────────────┘
           merge_rag_results reducer
           → {"rag_results": [r1, r2, r3]}
           → rag_synthesizer sees all 3 results
```

### Fallback When No Sub-Tasks

If the supervisor routes to `parallel_rag` but omits `rag_sub_tasks`, the planner node creates a single default task. `fan_out_rag_workers` would then emit a single `Send`, which executes as a standard single worker — functionally equivalent to the direct `rag_agent` path.

---

## 9. Design Patterns in This Codebase

### Pattern 1: Factory Functions (Node Factories)

**Where:** Every graph node except `parallel_planner_node`

```python
def make_supervisor_node(chat_llm, settings, callbacks, max_loops, registry):
    system_prompt = _build_supervisor_prompt(settings, registry)
    valid_agents = registry.valid_agent_names()
    loop_count = 0

    def supervisor_node(state: AgentState) -> Dict:
        nonlocal loop_count
        # ...
    return supervisor_node
```

**Why:** LangGraph nodes are simple `(state) -> dict` functions. Factory functions provide a way to:
1. Capture configuration (LLM, settings, callbacks) once at graph-build time
2. Pre-build expensive objects (system prompts, subgraphs) before the turn starts
3. Maintain per-graph mutable state (`loop_count`) via closure

**Contrast with class-based nodes:** A class instance `MyNode.__call__(self, state)` would work too, but factory functions are more composable and easier to test — you can call the factory with mock dependencies.

### Pattern 2: Supervisor Pattern

**Where:** `graph/supervisor.py`, `graph/builder.py`

The LLM-based supervisor implements the classic supervisor pattern for multi-agent systems:
- Central controller reads full context
- Decides which specialist to delegate to
- Loops until the task is complete (`__end__`)
- Has visibility into all prior specialist outputs (full message history)

```
supervisor → specialist A → supervisor → specialist B → supervisor → END
```

**Key design decision:** The supervisor does NOT call tools itself. It only routes. This keeps the supervisor's context focused on coordination, not execution.

**Compared to static routing:** A static router (e.g., keyword-based) would route once and never revisit. The supervisor can route to multiple specialists sequentially when a task requires it (e.g., compute something with utility_agent, then explain it using rag_agent).

### Pattern 3: Registry Pattern

**Where:** `agents/agent_registry.py`

`AgentRegistry` and `AgentSpec` implement a runtime catalog:
- Agents self-register via `AgentSpec` descriptors
- The supervisor's knowledge is derived from the registry at runtime
- New agents require: (1) register `AgentSpec`, (2) wire node in `builder.py`, (3) add route case in `route_from_supervisor`

**Benefit:** Adding `data_analyst` required no changes to `supervisor.py` — the supervisor prompt automatically included it via `{{available_agents}}` injection. The supervisor doesn't need to be edited for each new agent.

**Conditional registration:** `AgentSpec(enabled=docker_ok)` means Docker-unavailable deployments get a supervisor that simply never knows `data_analyst` exists.

### Pattern 4: ReAct Loop

**Where:** All specialist agents via `create_react_agent`

The ReAct (Reasoning + Acting) loop is the standard agentic pattern:
1. LLM **reasons** about what to do next (implicit in the text it generates)
2. LLM **acts** by emitting a tool call
3. System **observes** the tool result (ToolMessage)
4. Repeat until the LLM decides it has enough information

**Budget control:** `recursion_limit` prevents infinite loops. When the budget is exceeded, LangGraph raises `GraphRecursionError`, which is caught and returns a graceful "max steps exceeded" message.

### Pattern 5: Dependency Injection

**Where:** Throughout the entire codebase

All agents receive their dependencies explicitly:

```python
run_rag_agent(settings, stores, llm=chat_llm, judge_llm=judge_llm, session=session, ...)
make_utility_agent_node(chat_llm, settings, stores, session_proxy, callbacks)
```

No globals, no module-level singletons. This means:
- Easy to unit test (inject mock stores, mock LLMs)
- Multiple simultaneous tenants with different LLM configs are supported
- Settings changes at runtime are picked up on the next call

### Pattern 6: Strategy Pattern (Router Selection)

**Where:** `orchestrator.py` → `process_turn()`

```python
if self.ctx.settings.llm_router_enabled:
    decision = route_message_hybrid(...)   # strategy 1
else:
    decision = route_message(...)          # strategy 2
```

Both functions return the same `RouterDecision` type. The orchestrator doesn't care which strategy was used — only the result matters.

### Pattern 7: Proxy Pattern

**Where:** `graph/session_proxy.py`

`SessionProxy` is a structural proxy for `ChatSession`. Graph nodes cannot use `ChatSession` directly because:
1. `ChatSession` holds live state (message list, DB references)
2. LangGraph may serialize/deserialize state between nodes in some configurations
3. Passing `ChatSession` through `AgentState` would couple the graph to the session lifecycle

`SessionProxy` provides only the data that tools actually need (`scratchpad`, `session_id`, `tenant_id`, `uploaded_doc_ids`) as plain Python dataclass fields.

**Duck typing:** Tools never check `isinstance(session, ChatSession)` — they just access attributes. So `SessionProxy` works transparently without inheritance.

### Pattern 8: Contract Pattern

**Where:** `rag/agent.py` → `run_rag_agent()` return value

The RAG contract dict is a fixed-schema output:
```python
{
    "answer": str,
    "citations": List[citation_dict],
    "used_citation_ids": List[str],
    "confidence": float,
    "retrieval_summary": str,
    "warnings": List[str],
    "followups": List[str],
}
```

Every caller of `run_rag_agent()` knows this schema:
- `rag_node.py` calls `render_rag_contract(contract)`
- `orchestrator.py` calls `_render_rag_result(contract)`
- `rag_agent_tool.py` serializes it as JSON string
- `rag_worker_node.py` stores it in `rag_results`
- `rag_synthesizer_node.py` reads `contract["answer"]` and `contract["citations"]`

This allows the RAG internals to change (tool set, retrieval strategy, grading logic) without callers needing updates.

### Pattern 9: Fallback Chain

**Where:** `orchestrator._run_multi_agent_graph()` → `_is_graph_capability_error()` → `_run_general_agent_fallback()`

```
Primary:  multi-agent supervisor graph with specialist nodes
  ↓ (on capability/config error only)
Fallback: GeneralAgent ReAct loop with rag_agent_tool
  ↓ (on tool-calling not supported)
Fallback: plan-execute loop (LLM generates JSON plan, tools run sequentially)
```

`_is_graph_capability_error()` distinguishes expected capability mismatches from unexpected runtime errors. Only the former triggers fallback; the latter propagates and surfaces to the user.

### Pattern 10: Externalized Prompts

**Where:** `data/skills/*.md` + `rag/skills_loader.py`

Skill files are Markdown documents that serve as system prompts. Key design properties:
- **Hot-reload:** mtime-based cache means prompt changes take effect on the next request without restarting the app (for per-turn-loaded skills)
- **Template variables:** `{{available_agents}}` allows the prompt to be data-driven
- **Separation of concerns:** Prompt engineering is a separate activity from code changes; different files, different review process
- **Validation:** `REQUIRED_SECTIONS` ensures critical structural elements aren't accidentally removed

### Pattern 11: Sandbox Isolation

**Where:** `sandbox/docker_executor.py`

The Docker sandbox implements the principle of least privilege for code execution:
- **Network disabled:** `network_disabled=True` prevents exfiltration or SSRF
- **Memory cap:** `mem_limit="512m"` prevents memory exhaustion
- **Fresh container per call:** No state carries over between executions
- **Auto-remove:** `container.remove(force=True)` in `finally` block prevents container leak
- **No host FS access:** Files are copied via `put_archive()`; the host filesystem is not mounted

**Graceful degradation:** The entire `data_analyst` capability is disabled (not an error) when Docker is unavailable. `AgentRegistry._check_docker_available()` runs at graph build time and sets `enabled=False` on the `data_analyst` spec.

---

## 10. Class Reference Table

| Class | Module | Purpose | Key Methods / Attributes |
|---|---|---|---|
| `ChatbotApp` | `agents/orchestrator.py` | Main orchestrator for the turn lifecycle | `process_turn()`, `ingest_and_summarize_uploads()`, `_run_multi_agent_graph()` |
| `AppContext` | `agents/orchestrator.py` | Dependency container passed to `ChatbotApp` | `settings`, `providers`, `stores` |
| `ChatSession` | `agents/session.py` | Per-user conversation state (CLI/API) | `messages`, `scratchpad`, `uploaded_doc_ids`, `clear_scratchpad()` |
| `SessionProxy` | `graph/session_proxy.py` | Duck-type stand-in for `ChatSession` inside graph nodes | `scratchpad`, `session_id`, `tenant_id` |
| `AgentState` | `graph/state.py` | LangGraph state schema for the multi-agent graph | `messages` (reducer), `rag_results` (reducer), `next_agent`, `scratchpad` |
| `RAGResult` | `graph/state.py` | Output dataclass for a single parallel RAG worker | `query`, `doc_scope`, `contract`, `worker_id` |
| `AgentRegistry` | `agents/agent_registry.py` | Runtime catalog of available specialist agents | `register()`, `list_enabled()`, `valid_agent_names()`, `format_for_supervisor_prompt()` |
| `AgentSpec` | `agents/agent_registry.py` | Frozen descriptor for one specialist agent | `name`, `display_name`, `description`, `use_when`, `skills_key`, `enabled` |
| `RouterDecision` | `router/router.py` | Routing decision output from either router | `route`, `confidence`, `reasons`, `suggested_agent`, `router_method` |
| `SkillsLoader` | `rag/skills_loader.py` | mtime-cached markdown prompt loader with template substitution | `load(key, context)`, `invalidate(key)` |
| `DockerSandboxExecutor` | `sandbox/docker_executor.py` | Per-call isolated Python code execution in Docker | `execute(code, files, packages)` |
| `SandboxResult` | `sandbox/docker_executor.py` | Output of a sandbox execution | `stdout`, `stderr`, `exit_code`, `execution_time_seconds`, `truncated`, `success` |
| `SandboxUnavailableError` | `sandbox/exceptions.py` | Raised when Docker daemon is not reachable | — |
| `KnowledgeStores` | `rag/__init__.py` | Bundle of PostgreSQL store objects | `chunk_store`, `doc_store`, `mem_store` |
| `ProviderBundle` | `providers/llm_factory.py` | Bundle of LLM and embeddings providers | `chat`, `judge`, `embeddings` |
| `Settings` | `config.py` | Frozen dataclass of all env-var configuration | All settings fields; loaded via `load_settings()` |
| `LLMRouterOutput` | `router/llm_router.py` | Pydantic schema for structured LLM router output | `route`, `confidence`, `reasoning`, `suggested_agent` |
| `StateGraph` | `langgraph.graph` | LangGraph directed graph builder | `add_node()`, `add_edge()`, `add_conditional_edges()`, `compile()` |
| `Send` | `langgraph.types` | Deferred node invocation for parallel fan-out | `Send(node_name, state_dict)` |
| `MessagesState` | `langgraph.graph` | Base TypedDict with `messages: Annotated[list, add_messages]` | `messages` |
| `BaseChatModel` | `langchain_core.language_models` | Abstract base for all chat LLMs | `invoke()`, `bind_tools()`, `with_structured_output()` |
| `BaseMessage` | `langchain_core.messages` | Abstract base for all message types | `type`, `content`, `id` |
| `AIMessage` | `langchain_core.messages` | LLM response message | `content`, `tool_calls` |
| `HumanMessage` | `langchain_core.messages` | User input message | `content` |
| `SystemMessage` | `langchain_core.messages` | System prompt message | `content` |
| `ToolMessage` | `langchain_core.messages` | Tool execution result message | `content`, `tool_call_id` |

---

## How It All Connects — A Complete Request Trace

```
User: "Compare clause 5.2 in contract A with contract B"

1. ChatbotApp.process_turn()
   └── route_message_hybrid() → RouterDecision(route=AGENT, suggested_agent="")

2. _run_multi_agent_graph()
   └── AgentRegistry(settings)    # Docker available → 4 agents registered
   └── build_multi_agent_graph()  # StateGraph with 7 nodes compiled
   └── build_initial_state()      # {messages: [...], next_agent: "", ...}
   └── compiled_graph.invoke(state, config={callbacks, recursion_limit=50})

3. supervisor_node (loop 1)
   └── chat_llm.invoke([SystemMessage(prompt + available_agents), HumanMessage("Compare...")])
   └── LLM returns: {"next_agent": "parallel_rag", "rag_sub_tasks": [
         {"query": "clause 5.2 contract A", "preferred_doc_ids": ["doc_A"], "worker_id": "w0"},
         {"query": "clause 5.2 contract B", "preferred_doc_ids": ["doc_B"], "worker_id": "w1"}
       ]}
   └── state update: next_agent="parallel_rag", rag_sub_tasks=[task_A, task_B]

4. parallel_planner_node
   └── validates tasks, assigns worker_ids
   └── fan_out_rag_workers() returns [Send("rag_worker", state_A), Send("rag_worker", state_B)]

5. rag_worker (×2, concurrent)
   └── Worker 0: run_rag_agent(query="clause 5.2 contract A", preferred_doc_ids=["doc_A"])
   │     └── create_react_agent loop: search_chunks → extract_clause → scratchpad_write → ...
   │     └── returns RAG contract dict {answer, citations, confidence, ...}
   └── Worker 1: run_rag_agent(query="clause 5.2 contract B", preferred_doc_ids=["doc_B"])
         └── (same flow, isolated scratchpad)

6. merge_rag_results reducer
   └── [{worker_id: "w0", contract: {...}}, {worker_id: "w1", contract: {...}}]

7. rag_synthesizer_node
   └── chat_llm.invoke([SystemMessage(synthesis_prompt), ...both worker answers...])
   └── returns AIMessage(merged comparative answer with deduplicated citations)
   └── emits {"rag_results": [{"__clear__": True}]}

8. supervisor_node (loop 2)
   └── sees merged answer in messages
   └── LLM returns: {"next_agent": "__end__", "reasoning": "comparison complete"}

9. graph returns result
   └── final_answer = merged comparative answer
   └── session.messages synced

10. ChatbotApp returns text to user
```
