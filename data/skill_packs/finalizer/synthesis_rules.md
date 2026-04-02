# Finalizer Synthesis Rules
agent_scope: finalizer
task_tags: synthesis, citations, final_answer
description: Synthesis rules for composing the final user-facing answer from worker outputs.

## Finalization rules

- Preserve citations, caveats, and uncertainty from task results.
- Mention failed or incomplete tasks when they materially affect the answer.
- If task outputs conflict, surface the conflict instead of papering over it.
- If verification feedback is present, revise the answer to address it explicitly.
