"""Prompt templating system using Jinja2.

This module provides a flexible templating system for constructing
structured prompts with proper validation.
"""

from pathlib import Path
from typing import Any

from jinja2 import Template

from ragmark.exceptions import ConfigError
from ragmark.logger import get_logger

logger = get_logger(__name__)


class PromptTemplate:
    """Jinja2-based prompt template with validation.

    This class wraps Jinja2 templates and validates that all required
    variables are provided during rendering.

    Attributes:
        template: The compiled Jinja2 template.
        input_variables: List of required variable names.
    """

    def __init__(self, template: str, input_variables: list[str]):
        """Initialize the prompt template.

        Args:
            template: Jinja2 template string with {{ variable }} placeholders.
            input_variables: List of required variable names.

        Raises:
            ConfigError: If template syntax is invalid.
        """
        logger.debug(
            "Initializing prompt template: variables=%s, template_len=%d",
            input_variables,
            len(template),
        )

        try:
            self._jinja_template = Template(template)
        except Exception as e:
            logger.error("Invalid template syntax")
            logger.debug("Template error details: %s", e, exc_info=True)
            raise ConfigError(f"Invalid template syntax: {e}") from e

        self.input_variables = input_variables

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Template variables as keyword arguments.

        Returns:
            Rendered prompt string.

        Raises:
            ConfigError: If required variables are missing.
        """
        logger.debug("Rendering template: provided_vars=%s", list(kwargs.keys()))

        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            logger.error("Missing template variables: %s", missing)
            raise ConfigError(f"Missing required template variables: {missing}")

        try:
            rendered = self._jinja_template.render(**kwargs)
            logger.debug("Template rendered: output_len=%d", len(rendered))
            return rendered
        except Exception as e:
            logger.error("Template rendering failed")
            logger.debug("Rendering error details: %s", e, exc_info=True)
            raise ConfigError(f"Template rendering failed: {e}") from e

    @classmethod
    def from_file(cls, path: Path, input_variables: list[str]) -> "PromptTemplate":
        """Load template from a file.

        Args:
            path: Path to the template file.
            input_variables: List of required variable names.

        Returns:
            PromptTemplate instance.

        Raises:
            FileNotFoundError: If template file doesn't exist.
            ConfigError: If template is invalid.
        """
        logger.debug("Loading template from file: path=%s", path)

        if not path.exists():
            logger.error("Template file not found: path=%s", path)
            raise FileNotFoundError(f"Template file not found: {path}")

        template = path.read_text(encoding="utf-8")
        logger.info("Template loaded from file: path=%s, size=%d", path, len(template))

        return cls(template=template, input_variables=input_variables)


class StringPromptTemplate:
    """Simple string-based prompt template using Python .format().

    This class provides a lightweight alternative to Jinja2 templates,
    using standard Python string formatting. Ideal for simple prompts
    without complex logic.

    Attributes:
        template_str: The template string with {variable} placeholders.
        input_variables: List of required variable names.
    """

    def __init__(self, template_str: str, input_variables: list[str]):
        """Initialize the string prompt template.

        Args:
            template_str: Template string with {variable} placeholders.
            input_variables: List of required variable names.
        """
        self.template_str = template_str
        self.input_variables = input_variables

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Template variables as keyword arguments.

        Returns:
            Formatted prompt string.

        Raises:
            ConfigError: If required variables are missing or formatting fails.
        """
        logger.debug(
            "Formatting string template: provided_vars=%s", list(kwargs.keys())
        )

        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            logger.error("Missing template variables: %s", missing)
            raise ConfigError(f"Missing required template variables: {missing}")

        try:
            formatted = self.template_str.format(**kwargs)
            logger.debug("String template formatted: output_len=%d", len(formatted))
            return formatted
        except KeyError as e:
            logger.error("Template variable not found: %s", e)
            raise ConfigError(
                f"Template variable not found: {e}. "
                f"Required variables: {self.input_variables}"
            ) from e
        except Exception as e:
            logger.error("Template formatting failed")
            logger.debug("Formatting error details: %s", e, exc_info=True)
            raise ConfigError(f"Template formatting failed: {e}") from e

    @staticmethod
    def format_context(nodes: list[Any]) -> str:
        """Format a list of KnowledgeNode objects into a context string.

        This utility method takes retrieved nodes and formats them into
        a clean context string suitable for LLM prompts.

        Args:
            nodes: List of KnowledgeNode objects with .content attribute.

        Returns:
            Formatted context string with numbered chunks.
        """
        if not nodes:
            logger.debug("No nodes to format, returning empty context")
            return "(No context available)"

        logger.debug("Formatting context: node_count=%d", len(nodes))

        context_parts: list[str] = []
        for i, node in enumerate(nodes, start=1):
            content = node.node.content if hasattr(node, "node") else node.content

            context_parts.append(f"[{i}] {content.strip()}")

        formatted = "\n\n".join(context_parts)
        logger.debug("Context formatted: total_len=%d", len(formatted))

        return formatted

    @classmethod
    def from_file(
        cls, path: Path, input_variables: list[str]
    ) -> "StringPromptTemplate":
        """Load template from a file.

        Args:
            path: Path to the template file.
            input_variables: List of required variable names.

        Returns:
            StringPromptTemplate instance.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        logger.debug("Loading string template from file: path=%s", path)

        if not path.exists():
            logger.error("Template file not found: path=%s", path)
            raise FileNotFoundError(f"Template file not found: {path}")

        template_str = path.read_text(encoding="utf-8")
        logger.info(
            "String template loaded from file: path=%s, size=%d",
            path,
            len(template_str),
        )

        return cls(template_str=template_str, input_variables=input_variables)


RAG_QA_TEMPLATE = PromptTemplate(
    template="""System: You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{% for chunk in context_chunks %}
{{ chunk }}

{% endfor %}

User: {{ user_question }}
""",
    input_variables=["context_chunks", "user_question"],
)

RAG_QA_STRING_TEMPLATE = StringPromptTemplate(
    template_str="""You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

RAG_SUMMARIZE_TEMPLATE = PromptTemplate(
    template="""System: You are a summarization assistant. Create a concise summary based on the provided context.

Context:
{% for chunk in context_chunks %}
{{ chunk }}

{% endfor %}

User: Please summarize the above context in 2-3 sentences, highlighting the main points.
""",
    input_variables=["context_chunks"],
)

RAG_CHAT_TEMPLATE = PromptTemplate(
    template="""<|system|>
You are a helpful AI assistant. Answer questions based on the conversation history and provided context.

Context:
{% for chunk in context_chunks %}
{{ chunk }}

{% endfor %}
<|end|>

{% for message in chat_history %}
<|{{ message.role }}|>
{{ message.content }}
<|end|>
{% endfor %}

<|user|>
{{ user_question }}
<|end|>

<|assistant|>
""",
    input_variables=["context_chunks", "chat_history", "user_question"],
)
