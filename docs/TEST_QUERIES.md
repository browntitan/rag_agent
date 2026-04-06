# Test Queries

Use these prompts to smoke-test the live agent in this worktree.

## Basic chat

Source set: none beyond a working chat model.

- `Hello there`
- `What can you help me with in this application?`
- `Summarize the capabilities of this system in 5 bullets.`

Expected behavior:

- direct answer
- no citations required

## Grounded RAG

Source set: `data/kb/*`

- `Summarize the indexed MSA and cite the answer.`
- `Compare the indexed services agreement and cite the differences.`
- `What are the main requirements in the indexed policy docs? Cite your sources.`

Expected behavior:

- citations appear in the answer
- retrieval stays grounded in indexed KB docs

## Architecture docs

Source set: `docs/*.md` indexed through `KB_EXTRA_DIRS=./docs`

- `What are the key implementation details in the architecture docs? Cite your sources.`
- `Compare ARCHITECTURE.md and C4_ARCHITECTURE.md. Cite your sources.`
- `Explain the next-runtime foundation and how the service, kernel, query loop, and react executor fit together. Cite your sources.`

Expected behavior:

- citations from repo docs such as `ARCHITECTURE.md`, `C4_ARCHITECTURE.md`, and `NEXT_RUNTIME_FOUNDATION.md`
- no `No evidence available in the context` warning after KB sync
- when KB coverage is missing, the answer should tell you to run `python run.py sync-kb --collection-id default` instead of asking for pasted excerpts

## Upload and analyst flow

Source set: same-conversation uploaded files plus normal KB docs.

Upload:

- `new_demo_notebook/demo_data/regional_spend.csv`

Then ask:

- `Analyze the uploaded CSV and summarize the highest regional spend.`
- `What columns are in the uploaded file, and what do they represent?`
- `Give me 3 business insights from the uploaded CSV.`

Expected behavior:

- the file is attached to the same conversation
- analyst tools can see it from the session workspace

Regression prompt:

- start a fresh chat, then ask `Analyze the uploaded CSV and summarize the highest regional spend.`

Expected behavior:

- the app should explain that no uploaded datasets are attached to this conversation yet

## Coordinator and multi-step prompts

Source set: indexed KB plus runtime tools.

- `Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations.`
- `Find the most relevant documents for observability and explain how traces are stored locally versus in Langfuse. Cite evidence.`
- `Analyze the runtime design and explain when the coordinator should delegate to workers. Cite your sources.`

## Graceful failure prompts

- `Summarize the contents of the non-existent customer_churn_strategy.pdf and cite it.`
- `Use evidence from a file I have not uploaded yet and analyze it.`

Expected behavior:

- clear statement that the source is missing
- no fabricated citations
- guidance on what to upload or index first
