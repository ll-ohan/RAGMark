"""Unit tests for generation components.

This module tests prompt templating and LLM driver interfaces.
"""

from pathlib import Path

import pytest

from ragmark.exceptions import ConfigError
from ragmark.generation.prompts import RAG_QA_TEMPLATE, PromptTemplate


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_simple_template(self) -> None:
        """Test rendering a simple template."""
        template = PromptTemplate(
            template="Hello {{ name }}!",
            input_variables=["name"],
        )

        result = template.render(name="World")
        assert result == "Hello World!"

    def test_multiple_variables(self) -> None:
        """Test template with multiple variables."""
        template = PromptTemplate(
            template="{{ greeting }} {{ name }}, you are {{ age }} years old.",
            input_variables=["greeting", "name", "age"],
        )

        result = template.render(greeting="Hello", name="Alice", age=30)
        assert result == "Hello Alice, you are 30 years old."

    def test_missing_variable_raises_error(self) -> None:
        """Test that missing required variables raise ConfigError."""
        template = PromptTemplate(
            template="Hello {{ name }}!",
            input_variables=["name"],
        )

        with pytest.raises(ConfigError, match="Missing required template variables"):
            template.render()

    def test_extra_variables_allowed(self) -> None:
        """Test that extra variables are allowed."""
        template = PromptTemplate(
            template="Hello {{ name }}!",
            input_variables=["name"],
        )

        # Should not raise even with extra variable
        result = template.render(name="World", extra="ignored")
        assert result == "Hello World!"

    def test_jinja2_loops(self) -> None:
        """Test Jinja2 loop syntax."""
        template = PromptTemplate(
            template="Items: {% for item in items %}{{ item }}, {% endfor %}",
            input_variables=["items"],
        )

        result = template.render(items=["apple", "banana", "cherry"])
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    def test_jinja2_conditionals(self) -> None:
        """Test Jinja2 conditional syntax."""
        template = PromptTemplate(
            template="{% if show_greeting %}Hello {{ name }}{% endif %}",
            input_variables=["show_greeting", "name"],
        )

        result1 = template.render(show_greeting=True, name="Alice")
        assert result1 == "Hello Alice"

        result2 = template.render(show_greeting=False, name="Bob")
        assert result2 == ""

    def test_invalid_syntax_raises_error(self) -> None:
        """Test that invalid Jinja2 syntax raises ConfigError."""
        with pytest.raises(ConfigError, match="Invalid template syntax"):
            PromptTemplate(
                template="{{ unclosed",
                input_variables=[],
            )

    def test_from_file(self, tmp_path: Path) -> None:
        """Test loading template from file."""
        template_file = tmp_path / "template.txt"
        template_file.write_text("Hello {{ name }}!")

        template = PromptTemplate.from_file(
            template_file,
            input_variables=["name"],
        )

        result = template.render(name="World")
        assert result == "Hello World!"

    def test_from_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file(
                tmp_path / "nonexistent.txt",
                input_variables=[],
            )

    def test_rendering_error(self) -> None:
        """Test that rendering errors are caught and wrapped."""
        template = PromptTemplate(
            template="{{ items | length }}",
            input_variables=["items"],
        )

        # Passing a non-iterable should cause a rendering error
        with pytest.raises(ConfigError, match="Template rendering failed"):
            template.render(items=None)


class TestPredefinedTemplates:
    """Tests for pre-defined templates."""

    def test_rag_qa_template_structure(self) -> None:
        """Test that RAG_QA_TEMPLATE has correct structure."""
        assert isinstance(RAG_QA_TEMPLATE, PromptTemplate)
        assert "context_chunks" in RAG_QA_TEMPLATE.input_variables
        assert "user_question" in RAG_QA_TEMPLATE.input_variables

    def test_rag_qa_template_rendering(self) -> None:
        """Test rendering the RAG QA template."""
        result = RAG_QA_TEMPLATE.render(
            context_chunks=["Chunk 1", "Chunk 2"],
            user_question="What is the answer?",
        )

        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "What is the answer?" in result
        assert "System:" in result
        assert "Context:" in result
        assert "User:" in result
