"""Prompt templating system using Jinja2.

This module provides a flexible templating system for constructing
structured prompts with proper validation.
"""

from pathlib import Path
from typing import Any

from jinja2 import Template

from ragmark.exceptions import ConfigError


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
        try:
            self._jinja_template = Template(template)
        except Exception as e:
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
        # Validate all required variables are provided
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ConfigError(f"Missing required template variables: {missing}")

        try:
            return self._jinja_template.render(**kwargs)
        except Exception as e:
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
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        with open(path, encoding="utf-8") as f:
            template = f.read()

        return cls(template=template, input_variables=input_variables)


# Pre-defined templates
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
