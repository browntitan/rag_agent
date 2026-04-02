# Planner Task Decomposition
agent_scope: planner
task_tags: planning, decomposition, batching
description: How the planner should decompose work for the coordinator runtime.

## Planning rules

- Produce only the minimum number of tasks needed to complete the request.
- Use rag_worker for document-grounded work, utility for math or memory, data_analyst for tabular analysis, and general for bounded synthesis or general-purpose work.
- Mark tasks as parallel only when they are independent and can run without shared intermediate context.
- Put the actionable instruction in the input field so the worker brief is self-contained.
