# Agent Registry

The `AgentRegistry` is the single source of truth for which specialist agents the supervisor knows about. Instead of hardcoding agent descriptions inside the supervisor's skill file or Python code, the registry generates the supervisor's agent list dynamically and validates routing decisions at runtime.

---

## Why Dynamic Agent Discovery?

Without the registry, adding a new agent required editing:
1. `data/skills/supervisor_agent.md` — manually describe the new agent
2. `graph/supervisor.py` — add to `_VALID_AGENTS` hardcoded set
3. `graph/builder.py` — add node and edges

With the registry, step 1 is eliminated. The supervisor's prompt is rendered from the registry at node-build time. Steps 2 and 3 still require code changes (graph wiring is always explicit), but the prompt stays in sync automatically.

---

## Key Classes

### `AgentSpec`

A frozen dataclass describing one specialist agent:

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str          # Graph node name. e.g. "data_analyst"
    display_name: str  # Human-readable. e.g. "Data Analyst Agent"
    description: str   # 2-3 sentences shown in the supervisor prompt
    use_when: List[str]  # Bullet points: conditions for routing here
    skills_key: str    # SkillsLoader key for this agent's skill file
    enabled: bool = True  # False → hidden from supervisor entirely
```

`enabled=False` removes the agent from:
- The supervisor prompt (it never appears as an option)
- `valid_agent_names()` (the supervisor's response validator rejects routing to it)

### `AgentRegistry`

Manages the collection of `AgentSpec` objects:

```python
class AgentRegistry:
    def __init__(self, settings: Settings) -> None: ...

    def register(self, spec: AgentSpec) -> None: ...
    def get(self, name: str) -> AgentSpec | None: ...
    def list_enabled(self) -> List[AgentSpec]: ...
    def valid_agent_names(self) -> set[str]: ...
    def format_for_supervisor_prompt(self) -> str: ...
```

**`format_for_supervisor_prompt()`** renders a markdown block injected via `{{available_agents}}` into `data/skills/supervisor_agent.md`:

```markdown
## Available Agents

### 1. `rag_agent`
Searches, extracts, and reasons over documents ...

Use when:
- User asks about content in uploaded documents
- ...

### 2. `utility_agent`
...
```

**`valid_agent_names()`** returns the set of enabled agent names plus `"__end__"`. Used by the supervisor node to validate its JSON response and reject unknown agent names.

---

## Built-in Agents

| Name | Always Enabled? | Condition |
|---|---|---|
| `rag_agent` | Yes | Always |
| `utility_agent` | Yes | Always |
| `parallel_rag` | Yes | Always |
| `data_analyst` | Conditional | Requires Docker to be running |

The `data_analyst` agent is automatically disabled when `_check_docker_available()` returns `False` (Docker not reachable). A warning is logged. The agent can be re-enabled by starting Docker and restarting the app.

---

## How the Supervisor Uses the Registry

In `graph/supervisor.py`:

```python
def make_supervisor_node(chat_llm, settings, callbacks=None, max_loops=5, registry=None):
    if registry:
        valid_agents = registry.valid_agent_names()
        context = {"available_agents": registry.format_for_supervisor_prompt()}
    else:
        valid_agents = _VALID_AGENTS   # hardcoded fallback (backward compat)
        context = None

    system_prompt = load_supervisor_skills(settings, context=context)
    # {{available_agents}} in supervisor_agent.md is replaced with the registry markdown
```

In `agents/orchestrator.py`:

```python
registry = AgentRegistry(self._settings)
graph = build_multi_agent_graph(..., registry=registry)
```

---

## Adding a New Agent

Follow these 8 steps to add a new specialist agent called `my_agent`:

### Step 1 — Create tools file

```
src/agentic_chatbot/tools/my_agent_tools.py
```

Implement a factory function:
```python
def make_my_agent_tools(stores, session, *, settings) -> List[Any]:
    @tool
    def my_tool(arg: str) -> str:
        """Tool description."""
        ...
    return [my_tool]
```

### Step 2 — Create graph node

```
src/agentic_chatbot/graph/nodes/my_agent_node.py
```

Follow the pattern of `utility_node.py` or `data_analyst_node.py`:
```python
def make_my_agent_node(chat_llm, settings, stores, session_proxy, callbacks=None):
    from langgraph.prebuilt import create_react_agent
    from agentic_chatbot.tools.my_agent_tools import make_my_agent_tools
    from agentic_chatbot.rag.skills import load_my_agent_skills

    tools = make_my_agent_tools(stores, session_proxy, settings=settings)
    system_prompt = load_my_agent_skills(settings)
    agent_graph = create_react_agent(chat_llm, tools=tools)

    def my_agent_node(state):
        # ... same boilerplate as utility_node.py
        ...

    return my_agent_node
```

### Step 3 — Create skill file

```
data/skills/my_agent.md
```

Include `## Operating Rules` section (required by skills_loader validation).

### Step 4 — Add skills loader wrapper

In `src/agentic_chatbot/rag/skills.py`:
```python
def load_my_agent_skills(settings, *, context=None):
    return get_skills_loader(settings).load("my_agent", context=context)
```

### Step 5 — Add file mapping in skills_loader

In `src/agentic_chatbot/rag/skills_loader.py`:

In `_DEFAULTS`:
```python
"my_agent": "You are my_agent. Operating rules:\n1. ...\n",
```

In `_REQUIRED_SECTIONS`:
```python
"my_agent": ["Operating Rules"],
```

In `_get_path` mapping:
```python
"my_agent": getattr(s, "my_agent_skills_path", None),
```

In `config.py` Settings dataclass and `load_settings()`:
```python
my_agent_skills_path: Path  # skills_dir / "my_agent.md"
```

### Step 6 — Register in AgentRegistry

In `src/agentic_chatbot/agents/agent_registry.py`, inside `_register_builtin_agents()`:
```python
self.register(AgentSpec(
    name="my_agent",
    display_name="My Agent",
    description="Does my things.",
    use_when=["When user asks to do my thing"],
    skills_key="my_agent",
    enabled=True,  # or conditional on some check
))
```

### Step 7 — Wire into the graph

In `src/agentic_chatbot/graph/builder.py`:

```python
# In build_multi_agent_graph():
from agentic_chatbot.graph.nodes.my_agent_node import make_my_agent_node

my_agent_fn = make_my_agent_node(chat_llm, settings, stores, session_proxy, callbacks=callbacks)
graph.add_node("my_agent", my_agent_fn)
graph.add_edge("my_agent", "supervisor")
```

### Step 8 — Add route case

In `route_from_supervisor()` inside `builder.py`:
```python
elif next_agent == "my_agent":
    return "my_agent"
```

In the conditional edges map:
```python
"my_agent": "my_agent",
```

In `_VALID_SUGGESTED_AGENTS`:
```python
_VALID_SUGGESTED_AGENTS = {"rag_agent", "utility_agent", "parallel_rag", "data_analyst", "my_agent"}
```

---

## Configuration

The registry itself has no configuration. Individual agents may add their own env vars (see `DATA_ANALYST_AGENT.md` for an example). The `enabled` flag on each `AgentSpec` can be driven by a runtime check (like Docker availability) or by any other boolean condition.

---

## Where It Lives

| Responsibility | File |
|---|---|
| AgentSpec + AgentRegistry | `src/agentic_chatbot/agents/agent_registry.py` |
| Supervisor prompt injection | `data/skills/supervisor_agent.md` (uses `{{available_agents}}`) |
| Supervisor node integration | `src/agentic_chatbot/graph/supervisor.py` |
| Graph instantiation | `src/agentic_chatbot/agents/orchestrator.py` |
| Graph wiring | `src/agentic_chatbot/graph/builder.py` |
