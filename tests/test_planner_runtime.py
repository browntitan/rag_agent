from __future__ import annotations

from agentic_chatbot_next.runtime.task_plan import build_fallback_plan, normalise_task_plan, select_execution_batch


def test_single_step_request_produces_one_sequential_task():
    plan = build_fallback_plan("Summarize the latest uploaded policy.")

    assert len(plan) == 1
    assert plan[0]["mode"] == "sequential"
    assert plan[0]["executor"] == "rag_worker"


def test_comparison_request_produces_parallel_tasks():
    plan = build_fallback_plan('Compare "MSA v1" and "MSA v2" for indemnity differences.')

    assert len(plan) >= 2
    assert all(task["mode"] == "parallel" for task in plan)
    assert all(task["executor"] == "rag_worker" for task in plan)


def test_planner_routes_math_and_tabular_requests_to_specialists():
    utility_plan = build_fallback_plan("Calculate the monthly reserve from a 7% annual target.")
    analyst_plan = build_fallback_plan("Analyze this Excel spreadsheet and group revenue by region.")

    assert utility_plan[0]["executor"] == "utility"
    assert analyst_plan[0]["executor"] == "data_analyst"


def test_dependency_ordering_enforces_sequential_work():
    task_plan = normalise_task_plan(
        [
            {
                "id": "task_1",
                "title": "Gather requirements",
                "executor": "rag_worker",
                "mode": "sequential",
                "depends_on": [],
                "input": "Find the requirements.",
            },
            {
                "id": "task_2",
                "title": "Summarize findings",
                "executor": "general",
                "mode": "sequential",
                "depends_on": ["task_1"],
                "input": "Summarize the extracted requirements.",
            },
        ],
        query="Find the requirements and summarize them.",
        max_tasks=8,
    )

    first_batch = select_execution_batch(task_plan, [])
    second_batch = select_execution_batch(
        task_plan,
        [{"task_id": "task_1", "status": "completed", "output": "done"}],
    )

    assert [task["id"] for task in first_batch] == ["task_1"]
    assert [task["id"] for task in second_batch] == ["task_2"]
