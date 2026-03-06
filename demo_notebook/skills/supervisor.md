Supervisor routing preferences:
1) Route to `parallel_rag` when the user asks for compare/diff/synthesis across documents.
2) Route to `utility_agent` for list-doc or arithmetic tasks.
3) Route to `rag_agent` for policy/compliance/evidence requests.
4) End only when a complete answer is already present.
