from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set, Tuple
from urllib.parse import urlparse

from agentic_chatbot.config import Settings


@dataclass(frozen=True)
class DependencyIssue:
    module: str
    contexts: Tuple[str, ...]
    hint: str


@dataclass(frozen=True)
class ProviderConfigIssue:
    context: str
    message: str
    hint: str


_MODULE_HINTS: Mapping[str, str] = {
    "langchain_ollama": "python -m pip install langchain-ollama",
    "langchain_openai": "python -m pip install langchain-openai",
}


def _required_module_map(settings: Settings) -> Dict[str, Set[str]]:
    required: Dict[str, Set[str]] = {}

    provider_by_context = {
        "llm": settings.llm_provider.lower(),
        "judge": settings.judge_provider.lower(),
        "embeddings": settings.embeddings_provider.lower(),
    }

    for context, provider in provider_by_context.items():
        if provider == "ollama":
            required.setdefault("langchain_ollama", set()).add(context)
        elif provider == "azure":
            required.setdefault("langchain_openai", set()).add(context)

    return required


def validate_provider_dependencies(settings: Settings) -> List[DependencyIssue]:
    issues: List[DependencyIssue] = []
    required = _required_module_map(settings)

    for module_name, contexts in sorted(required.items()):
        if importlib.util.find_spec(module_name) is None:
            issues.append(
                DependencyIssue(
                    module=module_name,
                    contexts=tuple(sorted(contexts)),
                    hint=_MODULE_HINTS.get(module_name, f"python -m pip install {module_name}"),
                )
            )

    return issues


def _is_valid_azure_endpoint(value: str) -> bool:
    parsed = urlparse(value.strip())
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return host.endswith(".openai.azure.com") or host.endswith(".openai.azure.us")


def validate_provider_configuration(settings: Settings) -> List[ProviderConfigIssue]:
    issues: List[ProviderConfigIssue] = []
    provider_by_context = {
        "llm": settings.llm_provider.lower(),
        "judge": settings.judge_provider.lower(),
        "embeddings": settings.embeddings_provider.lower(),
    }

    if settings.ssl_cert_file and not settings.ssl_cert_file.exists():
        issues.append(
            ProviderConfigIssue(
                context="tls",
                message=f"SSL_CERT_FILE path does not exist: {settings.ssl_cert_file}",
                hint="Set SSL_CERT_FILE to a valid certificate bundle path or unset it.",
            )
        )

    uses_azure = "azure" in provider_by_context.values()
    if uses_azure and settings.azure_openai_endpoint and not _is_valid_azure_endpoint(settings.azure_openai_endpoint):
        issues.append(
            ProviderConfigIssue(
                context="azure",
                message=(
                    "AZURE_OPENAI_ENDPOINT must be an HTTPS Azure OpenAI endpoint. "
                    "Gov endpoints like https://<resource>.openai.azure.us are supported."
                ),
                hint="Set AZURE_OPENAI_ENDPOINT to your resource host (openai.azure.com or openai.azure.us).",
            )
        )
    if uses_azure:
        missing_shared: List[str] = []
        if not settings.azure_openai_api_key:
            missing_shared.append("AZURE_OPENAI_API_KEY")
        if not settings.azure_openai_endpoint:
            missing_shared.append("AZURE_OPENAI_ENDPOINT")
        if missing_shared:
            issues.append(
                ProviderConfigIssue(
                    context="azure",
                    message=f"Missing required Azure settings: {', '.join(missing_shared)}",
                    hint="Populate these shared Azure variables in .env before starting the app.",
                )
            )

    for context, provider in provider_by_context.items():
        if provider != "azure":
            continue

        deployment_name = {
            "llm": "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "judge": "AZURE_OPENAI_JUDGE_DEPLOYMENT",
            "embeddings": "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        }[context]
        deployment_value = {
            "llm": settings.azure_openai_chat_deployment,
            "judge": settings.azure_openai_judge_deployment,
            "embeddings": settings.azure_openai_embed_deployment,
        }[context]
        if not deployment_value:
            issues.append(
                ProviderConfigIssue(
                    context=context,
                    message=f"Missing required Azure deployment setting: {deployment_name}",
                    hint=f"Set {deployment_name} in .env for this provider context.",
                )
            )

    embed_provider = provider_by_context["embeddings"]
    embed_deployment = (settings.azure_openai_embed_deployment or "").lower()
    if embed_provider == "azure" and "ada-002" in embed_deployment and settings.embedding_dim != 1536:
        issues.append(
            ProviderConfigIssue(
                context="embeddings",
                message=(
                    "EMBEDDING_DIM must be 1536 for text-embedding-ada-002 deployments "
                    f"(current: {settings.embedding_dim})."
                ),
                hint="Set EMBEDDING_DIM=1536 and run `python run.py migrate-embedding-dim --yes` to rebuild vectors.",
            )
        )

    return issues


def format_dependency_issues(issues: Iterable[DependencyIssue]) -> str:
    rows = list(issues)
    if not rows:
        return "No provider dependency issues detected."

    lines: List[str] = ["Missing provider package dependencies detected:"]
    hints: Set[str] = set()

    for issue in rows:
        contexts = ", ".join(issue.contexts)
        lines.append(f"- {issue.module} (required by: {contexts})")
        hints.add(issue.hint)

    lines.extend(
        [
            "",
            "Recommended fixes:",
            "1) Install all project dependencies: python -m pip install -r requirements.txt",
        ]
    )

    next_step = 2
    for hint in sorted(hints):
        lines.append(f"{next_step}) Install missing provider package: {hint}")
        next_step += 1

    lines.extend(
        [
            f"{next_step}) If running via Docker, rebuild and restart app: docker compose up -d --build app",
            f"{next_step + 1}) Re-run preflight: python run.py doctor",
        ]
    )
    return "\n".join(lines)


def format_provider_config_issues(issues: Iterable[ProviderConfigIssue]) -> str:
    rows = list(issues)
    if not rows:
        return "No provider configuration issues detected."

    lines: List[str] = ["Provider configuration issues detected:"]
    hints: Set[str] = set()

    for issue in rows:
        lines.append(f"- ({issue.context}) {issue.message}")
        hints.add(issue.hint)

    lines.extend(
        [
            "",
            "Recommended fixes:",
            "1) Update your .env provider settings and restart the app.",
        ]
    )

    next_step = 2
    for hint in sorted(hints):
        lines.append(f"{next_step}) {hint}")
        next_step += 1

    lines.append(f"{next_step}) Re-run preflight: python run.py doctor")
    return "\n".join(lines)


class ProviderDependencyError(RuntimeError):
    def __init__(self, issues: Iterable[DependencyIssue]):
        self.issues: Tuple[DependencyIssue, ...] = tuple(issues)
        super().__init__(format_dependency_issues(self.issues))


class ProviderConfigurationError(RuntimeError):
    def __init__(self, issues: Iterable[ProviderConfigIssue]):
        self.issues: Tuple[ProviderConfigIssue, ...] = tuple(issues)
        super().__init__(format_provider_config_issues(self.issues))


def raise_if_missing_provider_dependencies(settings: Settings) -> None:
    issues = validate_provider_dependencies(settings)
    if issues:
        raise ProviderDependencyError(issues)


def raise_if_invalid_provider_configuration(settings: Settings) -> None:
    issues = validate_provider_configuration(settings)
    if issues:
        raise ProviderConfigurationError(issues)
