# Verifier Review Checklist
agent_scope: verifier
tool_tags: rag_agent_tool, list_indexed_docs, search_skills
task_tags: verification, review, critique
description: Checklist for deciding whether a final answer passes verification.

## Verification checklist

- Check whether the final answer overstates confidence beyond what the task outputs support.
- Check whether important caveats, missing evidence, or failed tasks were omitted.
- Check whether grounded claims are actually supported by the task artifacts and cited outputs.
- Return revise only when the final answer should materially change.
