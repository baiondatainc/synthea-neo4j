"""
LLM factory — returns the right LangChain chat model based on config.
Supports: Anthropic Claude, OpenAI, Ollama (local)
"""
from typing import Any
from config import get_settings


def get_llm(streaming: bool = False) -> Any:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            streaming=streaming,
            temperature=0,
            max_tokens=2048,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            temperature=0,
        )

    elif provider == "ollama":
        # Use ChatOllama (chat model) not OllamaLLM (completion model)
        # GraphCypherQAChain needs a chat model interface
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Set LLM_PROVIDER to 'anthropic', 'openai', or 'ollama'"
        )
