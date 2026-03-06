from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from agentic_chatbot.config import Settings
from agentic_chatbot.providers.dependency_checks import (
    raise_if_invalid_provider_configuration,
    raise_if_missing_provider_dependencies,
)


@dataclass(frozen=True)
class ProviderBundle:
    chat: object
    judge: object
    embeddings: object


def _build_httpx_client(settings: Settings):
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
    """Normalize OpenAI-compatible endpoint URLs to include /v1."""
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return value.strip()
    base = value.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def build_providers(settings: Settings) -> ProviderBundle:
    """Factory that builds the chat model, judge model, and embeddings.

    - Default: Ollama (ChatOllama + OllamaEmbeddings)
    - Optional: Azure OpenAI (AzureChatOpenAI + AzureOpenAIEmbeddings)
    - Optional: NVIDIA OpenAI-compatible endpoint for chat/judge (ChatOpenAI)

    We keep return types as `object` to avoid hard-coupling to a specific
    LangChain version/type hierarchy; callers typically treat these as
    Runnable chat models and Embeddings.
    """

    llm_provider = settings.llm_provider.lower()
    emb_provider = settings.embeddings_provider.lower()
    judge_provider = settings.judge_provider.lower()

    if llm_provider not in {"ollama", "azure", "nvidia"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    if emb_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")
    if judge_provider not in {"ollama", "azure", "nvidia"}:
        raise ValueError(f"Unsupported JUDGE_PROVIDER: {settings.judge_provider}")

    # Fail fast with actionable instructions before provider imports.
    raise_if_missing_provider_dependencies(settings)
    raise_if_invalid_provider_configuration(settings)
    http_client = _build_httpx_client(settings)

    # --- Chat model ---
    if llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            temperature=settings.ollama_temperature,
            num_predict=settings.ollama_num_predict,
            validate_model_on_init=True,
        )
    elif llm_provider == "azure":
        # Azure OpenAI
        from langchain_openai import AzureChatOpenAI

        chat = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=settings.azure_temperature,
            http_client=http_client,
        )
    else:
        from langchain_openai import ChatOpenAI

        base_url = _normalize_openai_base_url(settings.nvidia_openai_endpoint or "")
        nvidia_token = settings.nvidia_api_token or ""
        chat = ChatOpenAI(
            base_url=base_url,
            api_key="not-required",
            model=settings.nvidia_chat_model,
            temperature=settings.nvidia_temperature,
            http_client=http_client,
            default_headers={"Authorization": f"Bearer {nvidia_token}"},
        )

    # --- Judge model (can be provider/model-specific) ---
    if judge_provider == "ollama":
        from langchain_ollama import ChatOllama

        judge = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_judge_model,
            temperature=settings.judge_temperature,
            num_predict=settings.ollama_num_predict,
            validate_model_on_init=True,
        )
    elif judge_provider == "azure":
        from langchain_openai import AzureChatOpenAI
        judge = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_judge_deployment,
            temperature=settings.judge_temperature,
            http_client=http_client,
        )
    else:
        from langchain_openai import ChatOpenAI

        base_url = _normalize_openai_base_url(settings.nvidia_openai_endpoint or "")
        nvidia_token = settings.nvidia_api_token or ""
        judge = ChatOpenAI(
            base_url=base_url,
            api_key="not-required",
            model=settings.nvidia_judge_model,
            temperature=settings.nvidia_temperature,
            http_client=http_client,
            default_headers={"Authorization": f"Bearer {nvidia_token}"},
        )

    # --- Embeddings ---
    if emb_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
    else:
        from langchain_openai import AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embed_deployment,
            http_client=http_client,
            tiktoken_enabled=settings.tiktoken_enabled,
            check_embedding_ctx_length=settings.tiktoken_enabled,
        )

    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)
