# Utility Memory And Listing
agent_scope: utility
tool_tags: calculator, list_indexed_docs, memory_save, memory_load, memory_list
task_tags: utility, memory, listing
description: Safe usage rules for the utility agent tools.

## Utility rules

- Use the calculator for arithmetic and unit conversions instead of mental math.
- Use list_indexed_docs when document discovery is part of the task.
- Use memory tools only for user-confirmed durable facts, not speculative information.
- If a requested value is missing from memory, say that it is not stored instead of guessing.
