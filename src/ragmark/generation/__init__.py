"""LLM generation and prompt management.

This package provides abstractions for LLM drivers, prompt templating,
and context window management.
"""

from ragmark.generation.drivers import BaseLLMDriver, LlamaCppDriver
from ragmark.generation.prompts import (
    RAG_CHAT_TEMPLATE,
    RAG_QA_TEMPLATE,
    RAG_SUMMARIZE_TEMPLATE,
    PromptTemplate,
)

__all__ = [
    "BaseLLMDriver",
    "LlamaCppDriver",
    "PromptTemplate",
    "RAG_CHAT_TEMPLATE",
    "RAG_QA_TEMPLATE",
    "RAG_SUMMARIZE_TEMPLATE",
]
