from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class PreflightReport:
    checks: List[PreflightCheck] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "name": check.name,
                "ok": check.ok,
                "required": check.required,
                "detail": check.detail,
            }
            for check in self.checks
        ]


def run_preflight(
    *,
    repo_root: Path,
    runtime_root: Path,
    workspace_root: Path,
    memory_root: Path,
) -> PreflightReport:
    del repo_root
    checks: List[PreflightCheck] = []
    checks.extend(
        [
            _check_directory(runtime_root, "runtime_root"),
            _check_directory(workspace_root, "workspace_root"),
            _check_directory(memory_root, "memory_root"),
            _check_database(),
            _check_docker(),
        ]
    )
    checks.extend(_check_providers())
    return PreflightReport(checks=checks)


def _check_directory(path: Path, name: str) -> PreflightCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return PreflightCheck(name=name, ok=True, detail=str(path.resolve()))
    except Exception as exc:
        return PreflightCheck(name=name, ok=False, detail=f"{path}: {exc}")


def _check_database() -> PreflightCheck:
    dsn = os.getenv("PG_DSN", "postgresql://raguser:ragpass@localhost:5432/ragdb")
    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 5432)
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return PreflightCheck(name="database", ok=True, detail=f"{host}:{port} reachable")
    except Exception as exc:
        return PreflightCheck(name="database", ok=False, detail=f"{host}:{port} unreachable ({exc})")


def _check_docker() -> PreflightCheck:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return PreflightCheck(name="docker", ok=False, detail="docker binary not found in PATH")
    try:
        proc = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return PreflightCheck(name="docker", ok=False, detail=f"docker info failed ({exc})")
    if proc.returncode == 0:
        return PreflightCheck(name="docker", ok=True, detail="docker daemon reachable")
    stderr = (proc.stderr or proc.stdout or "").strip()[:240]
    return PreflightCheck(name="docker", ok=False, detail=f"docker daemon unavailable ({stderr})")


def _check_providers() -> List[PreflightCheck]:
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", llm_provider).lower()
    judge_provider = os.getenv("JUDGE_PROVIDER", llm_provider).lower()
    providers = {
        "chat_provider": llm_provider,
        "embeddings_provider": embeddings_provider,
        "judge_provider": judge_provider,
    }
    checks: List[PreflightCheck] = []
    checks.extend(_check_provider_role(name, provider) for name, provider in providers.items())
    if "ollama" in providers.values():
        checks.extend(_check_ollama_models(providers))
    return checks


def _check_provider_role(role_name: str, provider: str) -> PreflightCheck:
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
            if response.status_code == 200:
                return PreflightCheck(name=role_name, ok=True, detail=f"{provider} reachable at {base_url}")
            return PreflightCheck(name=role_name, ok=False, detail=f"{provider} returned HTTP {response.status_code} from {base_url}")
        except Exception as exc:
            return PreflightCheck(name=role_name, ok=False, detail=f"{provider} unreachable at {base_url} ({exc})")

    if provider == "azure":
        required = [
            ("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
            ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
        ]
    elif provider == "nvidia":
        required = [
            ("NVIDIA_API_TOKEN", os.getenv("NVIDIA_API_TOKEN")),
            ("NVIDIA_OPENAI_ENDPOINT", os.getenv("NVIDIA_OPENAI_ENDPOINT")),
        ]
    else:
        return PreflightCheck(name=role_name, ok=False, detail=f"Unsupported provider {provider!r} for notebook preflight")

    missing = [name for name, value in required if not value]
    if missing:
        return PreflightCheck(name=role_name, ok=False, detail=f"{provider} missing env vars: {', '.join(missing)}")
    return PreflightCheck(name=role_name, ok=True, detail=f"{provider} configuration present")


def _check_ollama_models(providers: Dict[str, str]) -> List[PreflightCheck]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = dict(response.json())
        models = {
            str(item.get("name") or "")
            for item in (payload.get("models") or [])
            if isinstance(item, dict)
        }
    except Exception as exc:
        return [PreflightCheck(name="ollama_models", ok=False, detail=f"could not inspect ollama models ({exc})")]

    checks: List[PreflightCheck] = []
    required_model_map = {
        "chat_provider": os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        "judge_provider": os.getenv("OLLAMA_JUDGE_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")),
        "embeddings_provider": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    }
    for role_name, provider in providers.items():
        if provider != "ollama":
            continue
        model_name = required_model_map[role_name]
        checks.append(
            PreflightCheck(
                name=f"{role_name}_model",
                ok=model_name in models,
                detail=f"{model_name} {'available' if model_name in models else 'missing'} at {base_url}",
            )
        )
    return checks
