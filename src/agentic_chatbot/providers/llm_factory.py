from __future__ import annotations

from dataclasses import dataclass

from agentic_chatbot.config import Settings


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

    if llm_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    if emb_provider not in {"ollama", "azure"}:
        raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")

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
        judge = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            temperature=0.0,
            num_predict=settings.ollama_num_predict,
            validate_model_on_init=True,
        )
    else:
        # Azure OpenAI
        from langchain_openai import AzureChatOpenAI

        if not (settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_chat_deployment):
            raise ValueError(
                "Azure selected but AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT missing"
            )

        chat = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=settings.azure_temperature,
        )
        judge = AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=0.0,
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

        if not (settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_embed_deployment):
            raise ValueError(
                "Azure embeddings selected but AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_EMBED_DEPLOYMENT missing"
            )
        embeddings = AzureOpenAIEmbeddings(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embed_deployment,
        )

    return ProviderBundle(chat=chat, judge=judge, embeddings=embeddings)
