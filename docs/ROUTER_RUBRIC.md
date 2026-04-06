# Router Rubric

The live router returns one of two routes:

- `BASIC`
- `AGENT`

## Live implementation

- deterministic rules: `src/agentic_chatbot_next/router/router.py`
- hybrid LLM escalation: `src/agentic_chatbot_next/router/llm_router.py`
- initial-agent selection: `src/agentic_chatbot_next/router/policy.py`

## Current agent hints

The router may suggest:

- `coordinator`
- `data_analyst`
- `rag_worker`
- `""`

If no hint is returned, AGENT turns normally start in `general`.
If `ENABLE_COORDINATOR_MODE=true`, `policy.py` forces `coordinator` as the initial AGENT
role regardless of the router hint.

## Hard AGENT signals

- attachments or uploads
- search / retrieval / citation requests
- document comparison
- high-stakes domains
- spreadsheet / CSV / pandas style requests
- clear multi-step workflows

## Typical BASIC signals

- greetings
- small talk
- lightweight general-knowledge questions
- conversational follow-ups that do not need tools
