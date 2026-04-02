# Data Analyst Dataset Workflow
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code, scratchpad_write, workspace_read
task_tags: data_analysis, pandas, sandbox
description: Standard workflow for dataset analysis in the sandboxed analyst environment.

## Analysis workflow

- Start with load_dataset to understand the available files and table shape.
- Use inspect_columns before writing code so you understand column types, nulls, and likely joins or aggregations.
- Write the plan to the scratchpad before longer code execution.
- Use execute_code for the actual pandas work and verify that the result answers the user's question before summarizing it.
