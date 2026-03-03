# Demo knowledge base packs

The repository ships with a longer `data/kb/` pack designed to exercise:

- retrieval across multiple documents
- chunk-level citations
- clause/requirement extraction workflows
- cross-document comparison and synthesis

## Pack A — Acme Agent Platform

- `01_product_overview.md`
- `02_pricing_and_plans.md`
- `03_security_and_privacy.md`
- `04_integrations_and_tools.md`
- `05_release_notes.md`

### Suggested prompts

- "According to the docs, what data retention policy applies to the Enterprise plan? Cite sections."
- "What tool schema best practices does the platform recommend?"
- "What changed between v1.3 and v1.4 that could affect tool calling?"

## Pack B — Engineering runbooks

- `runbook_incident_response.md`
- `runbook_oncall_handover.md`
- `runbook_data_pipeline.md`

### Suggested prompts

- "What is the update cadence during SEV-1 incidents? Cite the runbook."
- "Give me a checklist for oncall handover."

## Pack C — Internal API docs

- `api_auth.md`
- `api_endpoints.md`
- `api_rate_limits.md`
- `api_examples.md`

### Suggested prompts

- "How do I authenticate to the API? Provide the required headers and cite the docs."
- "What are the rate limits for the /v1/agents endpoints?"

## Upload demos

Try uploading one of your own files:

- PDF contract / spec
- Markdown meeting notes
- A long text file

Then ask:

- "Summarize the uploaded doc and list open questions."
- "Extract all key requirements and constraints with citations."
