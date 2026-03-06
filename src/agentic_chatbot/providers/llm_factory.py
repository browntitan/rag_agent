from __future__ import annotations

from dataclasses import dataclass

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


def build_providers(settings: Settings) -> ProviderBundle:
    """Factory that builds the chat model, judge model, and embeddings.

    - Default: Ollama (ChatOllama + OllamaEmbeddings)
    - Optional: Azure OpenAI (AzureChatOpenAI + AzureOpenAIEmbeddings)

    We keep return types as `object` to avoid hard-coupling to a specific
    LangChain version/type hierarchy; callers typically treat these as
    Runnable chat models and Embeddings.
    """

    llm_provider = settings.llm_provider.lower()
    emb_provider = settings.embeddings_provider.lower()
    judge_provider = settings.judge_provider.lower()

    if llm_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    if emb_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")
    if judge_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported JUDGE_PROVIDER: {settings.judge_provider}")

    # Fail fast with actionable instructions before provider imports.
    raise_if_missing_provider_dependencies(settings)
    raise_if_invalid_provider_configuration(settings)

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
    else:
        # Azure OpenAI
        from langchain_openai import AzureChatOpenAI

        chat = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=settings.azure_temperature,
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
    else:
        from langchain_openai import AzureChatOpenAI
        judge = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_judge_deployment,
            temperature=settings.judge_temperature,
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
        )

    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)
