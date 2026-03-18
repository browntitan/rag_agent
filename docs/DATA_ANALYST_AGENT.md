# Data Analyst Agent

The data analyst agent analyzes tabular data (Excel, CSV files) using Python pandas executed in a secure, isolated Docker sandbox. It follows a strict **plan-verify-reflect** workflow to produce reliable, auditable results.

---

## Overview

| Property | Value |
|---|---|
| Node name | `data_analyst` |
| Skill file | `data/skills/data_analyst_agent.md` |
| Tools | 7 (load_dataset, inspect_columns, execute_code, calculator, scratchpad ×3) |
| Sandbox | Docker container (network disabled, memory limited, auto-removed) |
| File source | Knowledge base only (files uploaded through the existing ingest pipeline) |
| Availability | Auto-disabled when Docker is not running on the host |

---

## Architecture: Plan-Verify-Reflect Workflow

The agent is instructed (via its skill file) to follow this mandatory 5-step workflow:

```
Step 1: Load & Inspect
  └─ load_dataset(doc_id)         → schema, shape, dtypes, first 5 rows
  └─ inspect_columns(doc_id, cols) → distributions, nulls, unique counts, stats
  └─ scratchpad_write("data_overview", "...")

Step 2: Plan
  └─ Think through approach: columns needed, null handling, dtype conversions
  └─ scratchpad_write("analysis_plan", "...")

Step 3: Execute
  └─ execute_code(code, doc_ids)  → runs in Docker sandbox, captures stdout

Step 4: Verify
  └─ Check: do results make sense? any errors in stderr?
  └─ Re-run with fixed code if needed

Step 5: Reflect & Respond
  └─ Summarize findings in natural language
  └─ Include numbers, trends, caveats, suggested follow-ups
```

This workflow prevents the agent from immediately writing code without understanding the data, which is the most common source of errors in data analysis tasks.

---

## Docker Sandbox

### Isolation properties

| Property | Value |
|---|---|
| Image | `python:3.12-slim` (configurable) |
| Network | Disabled (`network_disabled=True`) |
| Memory | 512 MiB limit (configurable) |
| Execution timeout | 60 seconds (configurable) |
| Container lifecycle | Created → files copied in → started → output captured → **always removed** |
| Output truncation | stdout and stderr each truncated to 50 KB |

### Pre-installed packages

Every execution automatically installs:
- `pandas` — dataframe operations
- `openpyxl` — `.xlsx` read/write
- `xlrd` — `.xls` legacy format

Additional packages can be specified but are not supported by default (to keep startup fast).

### File mounting

Files are copied into the container at `/workspace/<filename>` using the Docker SDK's `put_archive()` method. The agent's code should reference them as:

```python
df = pd.read_excel("/workspace/sales_data.xlsx")
df = pd.read_csv("/workspace/report.csv")
```

---

## Tools

### 1. `load_dataset(doc_id: str) -> str`

Load a dataset from the knowledge base. **Always call this first.**

**Args:**
- `doc_id`: Document ID from the knowledge base (use `list_indexed_docs` or `resolve_document` to find it)

**Returns JSON:**
```json
{
  "file_path": "/data/uploads/sales_q1.xlsx",
  "doc_id": "abc123",
  "columns": ["region", "revenue", "units"],
  "shape": [1200, 3],
  "dtypes": {"region": "object", "revenue": "float64", "units": "int64"},
  "head": [{"region": "North", "revenue": 45000.0, "units": 120}, ...],
  "info_summary": "1,200 rows x 3 columns"
}
```

**Side effect:** Stores the resolved file path in `session.scratchpad["dataset_{doc_id}"]` for other tools to reference.

**Error response:**
```json
{"error": "Document 'xyz' not found in knowledge base."}
```

---

### 2. `inspect_columns(doc_id: str, columns: str = "") -> str`

Get detailed per-column statistics. **Call this before writing analysis code.**

**Args:**
- `doc_id`: Must have been loaded via `load_dataset` first
- `columns`: Comma-separated column names. Leave empty for all columns.

**Returns JSON keyed by column name:**

For **numeric** columns:
```json
{
  "revenue": {
    "dtype": "float64",
    "count": 1200,
    "nulls": 3,
    "unique": 847,
    "mean": 42150.5,
    "std": 18230.2,
    "min": 1200.0,
    "p25": 28000.0,
    "p50": 40500.0,
    "p75": 55000.0,
    "max": 195000.0
  }
}
```

For **string/object** columns:
```json
{
  "region": {
    "dtype": "object",
    "count": 1200,
    "nulls": 0,
    "unique": 4,
    "top_values": {"North": 320, "South": 298, "East": 285, "West": 297}
  }
}
```

---

### 3. `execute_code(code: str, doc_ids: str = "") -> str`

Execute Python code in the Docker sandbox.

**Args:**
- `code`: Python source code. **Must use `print()` for output** — only stdout is captured.
- `doc_ids`: Comma-separated doc_ids whose files to mount at `/workspace/`.

**Example code:**
```python
import pandas as pd

df = pd.read_csv("/workspace/sales.csv")
result = df.groupby("region")["revenue"].mean().round(2)
print(result.to_string())
```

**Returns JSON:**
```json
{
  "stdout": "region\nEast     41230.50\nNorth    43100.25\nSouth    40890.75\nWest     43480.00",
  "stderr": "",
  "success": true,
  "execution_time_seconds": 1.23,
  "truncated": false
}
```

**Error response (Docker unavailable):**
```json
{
  "error": "Docker sandbox is not available: ...",
  "success": false,
  "stdout": "",
  "stderr": ""
}
```

---

### 4. `calculator(expression: str) -> str`

Safe arithmetic evaluation for simple math. Avoids spinning up a Docker container for trivial calculations.

**Use for:** percentages, unit conversions, basic formulas.

**Example:** `calculator("42000 * 1.15")` → `"48300.0"`

---

### 5-7. Scratchpad Tools

Within-turn memory for tracking observations, plans, and intermediate findings.

| Tool | Purpose |
|---|---|
| `scratchpad_write(key, value)` | Save a finding or plan |
| `scratchpad_read(key)` | Retrieve a saved value |
| `scratchpad_list()` | List all saved keys |

**Recommended usage pattern:**
```
scratchpad_write("data_overview", "1200 rows, 3 cols. revenue has 3 nulls.")
scratchpad_write("analysis_plan", "1. Drop nulls. 2. Group by region. 3. Mean revenue.")
```

---

## File Flow

```
User uploads file (e.g. sales.xlsx)
  → ingest_paths() stores file on disk + metadata in doc_store
  → doc_store.source_path = "/data/uploads/sales.xlsx"

User asks: "What is the average revenue by region?"
  → supervisor routes to data_analyst
  → agent calls load_dataset("abc123")
      → doc_store.get_document("abc123") → source_path
      → session.scratchpad["dataset_abc123"] = "/data/uploads/sales.xlsx"
  → agent calls execute_code(code, doc_ids="abc123")
      → files = {"/workspace/sales.xlsx": "/data/uploads/sales.xlsx"}
      → DockerSandboxExecutor.execute(code, files)
          → container.put_archive("/workspace/", tar_of_sales.xlsx)
          → code runs: pd.read_excel("/workspace/sales.xlsx")
          → stdout captured, container removed
      → result returned to agent
  → agent summarizes findings in natural language
```

---

## Configuration

All settings are environment variables with defaults:

| Env Var | Default | Description |
|---|---|---|
| `SANDBOX_DOCKER_IMAGE` | `python:3.12-slim` | Docker image for the sandbox |
| `SANDBOX_TIMEOUT_SECONDS` | `60` | Max execution time per code run |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Container memory limit |
| `DATA_ANALYST_MAX_STEPS` | `10` | Max LLM turns in the ReAct loop |
| `DATA_ANALYST_SKILLS_PATH` | `{SKILLS_DIR}/data_analyst_agent.md` | Skill file path |

---

## Error Handling

### Docker not available at startup
- `AgentRegistry._check_docker_available()` pings Docker on init
- If Docker is unreachable, `data_analyst` is registered with `enabled=False`
- The supervisor never receives `data_analyst` in its prompt and will not route to it
- Queries about data analysis will fall through to `utility_agent` or `rag_agent`

### Code execution errors
- `execute_code` returns `success: false` with the stderr traceback
- The agent reads the error, adjusts the code, and retries (up to budget)

### Timeout
- Container is killed immediately
- `execute_code` returns `success: false`, `exit_code: -1`, and a timeout message
- Agent may attempt a simpler/faster version of the code

### Missing file
- `load_dataset` returns an error JSON if the file path is not on disk
- Typically indicates a stale ingest record; user should re-upload

---

## Security

| Property | Detail |
|---|---|
| Network | Completely disabled inside sandbox |
| Filesystem | Read-only root except `/workspace/` |
| Memory | Hard-limited (default 512 MiB) |
| Execution time | Hard-limited (default 60s) |
| Container reuse | None — fresh container per `execute_code` call |
| Cleanup | Container always removed in `finally` block |
| Output | Truncated at 50 KB to prevent memory exhaustion |

---

## Example Interaction

**User:** "What is the average revenue by region in the Q1 sales file?"

**Agent internal flow:**

1. `load_dataset("sales_q1")` → columns: `[region, revenue, units]`, 1200 rows
2. `scratchpad_write("data_overview", "1200 rows x 3 cols. revenue is float64, 3 nulls.")`
3. `inspect_columns("sales_q1", "revenue,region")` → revenue: mean=42150, region: 4 unique values
4. `scratchpad_write("analysis_plan", "Drop 3 null revenue rows. Group by region. Compute mean.")`
5. `execute_code("import pandas as pd\ndf = pd.read_csv('/workspace/sales_q1.csv')\ndf = df.dropna(subset=['revenue'])\nprint(df.groupby('region')['revenue'].mean().round(2).to_string())", "sales_q1")`
6. Verify stdout: numbers are in expected range ✓
7. Respond with natural language summary

**User sees:**

> The average revenue by region in the Q1 sales file is:
> - **North:** $43,100
> - **East:** $41,231
> - **South:** $40,891
> - **West:** $43,480
>
> Note: 3 rows with missing revenue values were excluded from the calculation. The North and West regions performed similarly and both outperformed East and South by ~5%.
