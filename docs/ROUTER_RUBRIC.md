# Router rubric

The router produces one of two routes:

- `BASIC` — plain LLM call (`run_basic_chat`), no tools.
- `AGENT` — multi-agent supervisor graph (with fallback to legacy `GeneralAgent` if graph setup fails).

The router is intentionally deterministic and cheap.

## Inputs used

- `user_text`
- `has_attachments` (bool)
- `explicit_force_agent` (bool, from CLI `--force-agent`)

## Hard rules (always AGENT)

1) Attachments present
- Upload turns should use the agent path and grounding/citations.

2) Tool / multi-step verbs
- e.g. "search", "look up", "calculate", "open file", "summarize this PDF", "compare and recommend"

3) Citations / grounding requested
- e.g. "cite", "sources", "according to", "evidence", "grounded"

4) High-stakes topic hints
- legal, medical, financial, security/compliance

## Additional heuristic

- Very long input (`len(user_text) > 600`) routes to `AGENT`.

## Prefer BASIC when

- no attachments
- no tool-like intent
- low complexity, general knowledge / conversational

## Where this is implemented

See `src/agentic_chatbot/router/router.py`.
