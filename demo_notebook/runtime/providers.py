from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings

from .config import NotebookSettings


@dataclass(frozen=True)
class ProviderBundle:
    chat: object
    judge: object
    embeddings: Embeddings


class LocalHashEmbeddings(Embeddings):
    """Simple deterministic fallback embeddings for notebook demos.

    This is only used for vLLM when an embeddings endpoint is unavailable.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        tokens = text.lower().split()
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = -1.0 if digest[4] % 2 else 1.0
            mag = (digest[5] / 255.0) + 0.5
            vec[idx] += sign * mag

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def _build_httpx_client(settings: NotebookSettings):
    import httpx

    verify: bool | str = True
    if not settings.ssl_verify:
        verify = False
    elif settings.ssl_cert_file:
        verify = str(settings.ssl_cert_file)

    return httpx.Client(
        http2=settings.http2_enabled,
        verify=verify,
        timeout=httpx.Timeout(60.0, connect=20.0),
    )


def _normalize_openai_base_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return value.strip()
    base = value.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _build_azure(settings: NotebookSettings) -> ProviderBundle:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    missing = []
    if not settings.azure_api_key:
        missing.append("NOTEBOOK_AZURE_API_KEY")
    if not settings.azure_endpoint:
        missing.append("NOTEBOOK_AZURE_ENDPOINT")
    if not settings.azure_chat_deployment:
        missing.append("NOTEBOOK_AZURE_CHAT_DEPLOYMENT")
    if not settings.azure_judge_deployment:
        missing.append("NOTEBOOK_AZURE_JUDGE_DEPLOYMENT")
    if not settings.azure_embed_deployment:
        missing.append("NOTEBOOK_AZURE_EMBED_DEPLOYMENT")
    if missing:
        raise ValueError(f"Azure provider missing required settings: {', '.join(missing)}")

    http_client = _build_httpx_client(settings)

    chat = AzureChatOpenAI(
        api_key=settings.azure_api_key,
        azure_endpoint=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        azure_deployment=settings.azure_chat_deployment,
        temperature=settings.temperature,
        http_client=http_client,
    )
    judge = AzureChatOpenAI(
        api_key=settings.azure_api_key,
        azure_endpoint=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        azure_deployment=settings.azure_judge_deployment,
        temperature=settings.judge_temperature,
        http_client=http_client,
    )
    embeddings = AzureOpenAIEmbeddings(
        api_key=settings.azure_api_key,
        azure_endpoint=settings.azure_endpoint,
        api_version=settings.azure_api_version,
        azure_deployment=settings.azure_embed_deployment,
        http_client=http_client,
        tiktoken_enabled=settings.tiktoken_enabled,
        check_embedding_ctx_length=settings.tiktoken_enabled,
    )
    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)


def _build_ollama(settings: NotebookSettings) -> ProviderBundle:
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    chat = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.temperature,
        num_predict=settings.ollama_num_predict,
    )
    judge = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_judge_model,
        temperature=settings.judge_temperature,
        num_predict=settings.ollama_num_predict,
    )
    embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )
    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)


def _build_vllm(settings: NotebookSettings) -> ProviderBundle:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not settings.vllm_base_url:
        raise ValueError("NOTEBOOK_VLLM_BASE_URL is required when NOTEBOOK_PROVIDER=vllm")
    if not settings.vllm_chat_model:
        raise ValueError("NOTEBOOK_VLLM_CHAT_MODEL is required when NOTEBOOK_PROVIDER=vllm")

    base_url = settings.vllm_base_url.rstrip("/")
    base_url = _normalize_openai_base_url(base_url)
    http_client = _build_httpx_client(settings)

    chat = ChatOpenAI(
        base_url=base_url,
        api_key=settings.vllm_api_key or "not-required",
        model=settings.vllm_chat_model,
        temperature=settings.temperature,
        http_client=http_client,
    )

    judge_model = settings.vllm_judge_model or settings.vllm_chat_model
    judge = ChatOpenAI(
        base_url=base_url,
        api_key=settings.vllm_api_key or "not-required",
        model=judge_model,
        temperature=settings.judge_temperature,
        http_client=http_client,
    )

    if settings.vllm_use_openai_embeddings and settings.vllm_embed_model:
        embeddings = OpenAIEmbeddings(
            base_url=base_url,
            api_key=settings.vllm_api_key or "not-required",
            model=settings.vllm_embed_model,
            http_client=http_client,
            tiktoken_enabled=settings.tiktoken_enabled,
            check_embedding_ctx_length=settings.tiktoken_enabled,
        )
    else:
        embeddings = LocalHashEmbeddings(dim=settings.embedding_dim)

    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)


def _build_nvidia(settings: NotebookSettings) -> ProviderBundle:
    from langchain_ollama import OllamaEmbeddings
    from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI

    if not settings.nvidia_endpoint:
        raise ValueError("NOTEBOOK_NVIDIA_ENDPOINT is required when NOTEBOOK_PROVIDER=nvidia")
    if not settings.nvidia_token:
        raise ValueError("NOTEBOOK_NVIDIA_TOKEN (or legacy Token) is required when NOTEBOOK_PROVIDER=nvidia")
    if not settings.nvidia_chat_model:
        raise ValueError("NOTEBOOK_NVIDIA_CHAT_MODEL is required when NOTEBOOK_PROVIDER=nvidia")
    if not settings.nvidia_judge_model:
        raise ValueError("NOTEBOOK_NVIDIA_JUDGE_MODEL is required when NOTEBOOK_PROVIDER=nvidia")

    base_url = _normalize_openai_base_url(settings.nvidia_endpoint)
    http_client = _build_httpx_client(settings)
    auth_headers = {"Authorization": f"Bearer {settings.nvidia_token}"}

    chat = ChatOpenAI(
        base_url=base_url,
        api_key="not-required",
        model=settings.nvidia_chat_model,
        temperature=settings.nvidia_temperature,
        http_client=http_client,
        default_headers=auth_headers,
    )

    judge = ChatOpenAI(
        base_url=base_url,
        api_key="not-required",
        model=settings.nvidia_judge_model,
        temperature=settings.nvidia_temperature,
        http_client=http_client,
        default_headers=auth_headers,
    )

    backend = settings.nvidia_embeddings_backend
    if backend == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    elif backend == "azure":
        missing = []
        if not settings.azure_api_key:
            missing.append("NOTEBOOK_AZURE_API_KEY")
        if not settings.azure_endpoint:
            missing.append("NOTEBOOK_AZURE_ENDPOINT")
        if not settings.azure_embed_deployment:
            missing.append("NOTEBOOK_AZURE_EMBED_DEPLOYMENT")
        if missing:
            raise ValueError(
                "NOTEBOOK_NVIDIA_EMBEDDINGS_BACKEND=azure requires: " + ", ".join(missing)
            )
        embeddings = AzureOpenAIEmbeddings(
            api_key=settings.azure_api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
            azure_deployment=settings.azure_embed_deployment,
            http_client=http_client,
            tiktoken_enabled=settings.tiktoken_enabled,
            check_embedding_ctx_length=settings.tiktoken_enabled,
        )
    else:
        embeddings = LocalHashEmbeddings(dim=settings.embedding_dim)

    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)


def build_provider_bundle(settings: NotebookSettings) -> ProviderBundle:
    mode = settings.provider_mode.lower()
    if mode == "azure":
        return _build_azure(settings)
    if mode == "ollama":
        return _build_ollama(settings)
    if mode == "vllm":
        return _build_vllm(settings)
    if mode == "nvidia":
        return _build_nvidia(settings)
    raise ValueError(f"Unsupported NOTEBOOK_PROVIDER={settings.provider_mode!r}")
