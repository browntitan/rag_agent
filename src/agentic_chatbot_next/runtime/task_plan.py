from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

_VALID_EXECUTORS = {
    "rag_worker",
    "utility",
    "data_analyst",
    "general",
}
_VALID_MODES = {"sequential", "parallel"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "stopped"}


@dataclass
class TaskSpec:
    id: str
    title: str
    executor: str
    mode: str
    depends_on: List[str] = field(default_factory=list)
    input: str = ""
    doc_scope: List[str] = field(default_factory=list)
    skill_queries: List[str] = field(default_factory=list)
    status: str = "pending"
    artifact_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "executor": self.executor,
            "mode": self.mode,
            "depends_on": list(self.depends_on),
            "input": self.input,
            "doc_scope": list(self.doc_scope),
            "skill_queries": list(self.skill_queries),
            "status": self.status,
            "artifact_ref": self.artifact_ref or f"task:{self.id}",
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, index: int = 0) -> "TaskSpec":
        task_id = str(raw.get("id") or f"task_{index + 1}")
        title = str(raw.get("title") or f"Task {index + 1}")
        executor = str(raw.get("executor") or "general").strip()
        if executor not in _VALID_EXECUTORS:
            executor = _infer_executor(f"{title}\n{raw.get('input', '')}")
        mode = str(raw.get("mode") or "sequential").strip().lower()
        if mode not in _VALID_MODES:
            mode = "sequential"
        depends_on = [str(item) for item in (raw.get("depends_on") or []) if str(item)]
        doc_scope = [str(item) for item in (raw.get("doc_scope") or []) if str(item)]
        skill_queries = [str(item) for item in (raw.get("skill_queries") or []) if str(item)]
        status = str(raw.get("status") or "pending")
        artifact_ref = str(raw.get("artifact_ref") or f"task:{task_id}")
        return cls(
            id=task_id,
            title=title,
            executor=executor,
            mode=mode,
            depends_on=depends_on,
            input=str(raw.get("input") or ""),
            doc_scope=doc_scope,
            skill_queries=skill_queries,
            status=status,
            artifact_ref=artifact_ref,
        )


@dataclass
class TaskResult:
    task_id: str
    title: str
    executor: str
    status: str
    output: str
    artifact_ref: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "executor": self.executor,
            "status": self.status,
            "output": self.output,
            "artifact_ref": self.artifact_ref,
            "warnings": list(self.warnings),
        }


@dataclass
class WorkerExecutionRequest:
    agent_name: str
    task_id: str
    title: str
    prompt: str
    description: str = ""
    doc_scope: List[str] = field(default_factory=list)
    skill_queries: List[str] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "title": self.title,
            "prompt": self.prompt,
            "description": self.description,
            "doc_scope": list(self.doc_scope),
            "skill_queries": list(self.skill_queries),
            "artifact_refs": list(self.artifact_refs),
            "metadata": dict(self.metadata),
        }


@dataclass
class VerificationResult:
    status: str = "pass"
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "issues": list(self.issues),
            "feedback": self.feedback,
        }


@dataclass
class TaskExecutionState:
    user_request: str
    planner_summary: str
    task_plan: List[Dict[str, Any]] = field(default_factory=list)
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    partial_answer: str = ""
    final_answer: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_request": self.user_request,
            "planner_summary": self.planner_summary,
            "task_plan": [dict(item) for item in self.task_plan],
            "task_results": [dict(item) for item in self.task_results],
            "partial_answer": self.partial_answer,
            "final_answer": self.final_answer,
            "verification": dict(self.verification),
        }


def _infer_executor(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("csv", "excel", "spreadsheet", "dataframe", "pandas")):
        return "data_analyst"
    if (
        any(token in lower for token in ("calculate", "math", "convert", "memory", "remember"))
        or re.search(r"\b(sum|difference|percent|percentage|total|average|mean)\b", lower)
    ):
        return "utility"
    if any(
        token in lower
        for token in ("document", "contract", "policy", "clause", "requirement", "kb", "knowledge base", "upload", "compare")
    ):
        return "rag_worker"
    return "general"


def _extract_doc_hints(query: str) -> List[str]:
    hints = re.findall(r'"([^"]+)"', query)
    if hints:
        return [hint.strip() for hint in hints if hint.strip()]
    matches = re.findall(r"\b(?:doc(?:ument)?|contract|policy|runbook|file)\s+([a-z0-9._-]+)", query, flags=re.I)
    return [match.strip() for match in matches if match.strip()]


def build_fallback_plan(query: str, *, max_tasks: int = 8) -> List[Dict[str, Any]]:
    lower = query.lower()
    doc_hints = _extract_doc_hints(query)

    if any(token in lower for token in ("compare", "diff", "difference")) and len(doc_hints) >= 2:
        tasks: List[TaskSpec] = []
        for index, hint in enumerate(doc_hints[: max_tasks - 1], start=1):
            task_id = f"task_{index}"
            tasks.append(
                TaskSpec(
                    id=task_id,
                    title=f"Analyze {hint}",
                    executor="rag_worker",
                    mode="parallel",
                    input=f"Analyze the document or source '{hint}' for the user's comparison request: {query}",
                    doc_scope=[hint],
                    skill_queries=[
                        "document resolution and ambiguity handling",
                        "multi-document comparison workflows",
                        "citation hygiene and synthesis rules",
                    ],
                )
            )
        return [task.to_dict() for task in tasks[:max_tasks]]

    executor = _infer_executor(query)
    skill_queries: List[str] = []
    if executor == "rag_worker":
        skill_queries = [
            "document resolution and ambiguity handling",
            "retrieval strategy selection",
            "citation hygiene and synthesis rules",
        ]
    elif executor == "data_analyst":
        skill_queries = ["dataset inspection", "analysis planning", "safe code execution"]
    elif executor == "utility":
        skill_queries = ["calculator usage", "memory recall", "document listing"]

    return [
        TaskSpec(
            id="task_1",
            title="Handle user request",
            executor=executor,
            mode="sequential",
            input=query,
            doc_scope=doc_hints,
            skill_queries=skill_queries,
        ).to_dict()
    ]


def normalise_task_plan(raw_tasks: Any, *, query: str, max_tasks: int) -> List[Dict[str, Any]]:
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
    if not tasks:
        return build_fallback_plan(query, max_tasks=max_tasks)
    normalised = [
        TaskSpec.from_dict(task, index=index).to_dict()
        for index, task in enumerate(tasks[:max_tasks])
        if isinstance(task, dict)
    ]
    return normalised or build_fallback_plan(query, max_tasks=max_tasks)


def completed_task_ids(task_results: List[Dict[str, Any]]) -> set[str]:
    return {
        str(result.get("task_id"))
        for result in task_results
        if isinstance(result, dict) and str(result.get("status")) == "completed"
    }


def attempted_task_ids(task_results: List[Dict[str, Any]]) -> set[str]:
    return {
        str(result.get("task_id"))
        for result in task_results
        if isinstance(result, dict) and str(result.get("task_id"))
    }


def select_execution_batch(
    task_plan: List[Dict[str, Any]],
    task_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    done = completed_task_ids(task_results)
    attempted = attempted_task_ids(task_results)

    ready: List[Dict[str, Any]] = []
    for task in task_plan:
        task_id = str(task.get("id", ""))
        if not task_id or task_id in attempted:
            continue
        dependencies = [str(dep) for dep in (task.get("depends_on") or []) if str(dep)]
        if any(dep not in done for dep in dependencies):
            continue
        ready.append(task)

    if not ready:
        return []

    first = ready[0]
    if str(first.get("mode", "sequential")) != "parallel":
        return [first]

    batch: List[Dict[str, Any]] = []
    for task in ready:
        if str(task.get("mode", "sequential")) != "parallel":
            break
        batch.append(task)
    return batch
