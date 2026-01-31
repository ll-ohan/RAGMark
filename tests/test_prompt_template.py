"""Unit and integration tests for prompt templating logic."""

from collections.abc import Callable
from pathlib import Path

import pytest
from jinja2 import TemplateSyntaxError

from ragmark.exceptions import ConfigError
from ragmark.generation.prompts import PromptTemplate, StringPromptTemplate
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import RetrievedNode


class TestStringPromptTemplate:
    """Test suite for StringPromptTemplate logic and robustness (f-string based)."""

    @pytest.mark.unit
    def test_format_should_render_correctly_with_valid_inputs(self) -> None:
        """Verify standard formatting success without file I/O.

        Given:
            A valid StringPromptTemplate and matching input variables.
        When:
            The format method is called.
        Then:
            The output string must be correctly interpolated.
        """
        template = StringPromptTemplate(
            template_str="Hello {name}", input_variables=["name"]
        )

        result = template.format(name="World")

        assert result == "Hello World"

    @pytest.mark.unit
    def test_format_should_raise_config_error_on_missing_variable(self) -> None:
        """Verify that validation errors (missing input) do not have an underlying cause.

        Given:
            A StringPromptTemplate requiring the 'name' variable.
        When:
            The format method is called without any arguments.
        Then:
            A ConfigError is raised, and its __cause__ must be None.
        """
        template = StringPromptTemplate(
            template_str="Hello {name}", input_variables=["name"]
        )

        with pytest.raises(
            ConfigError, match="Missing required template variables"
        ) as exc_info:
            template.format()

        assert exc_info.value.__cause__ is None

    @pytest.mark.unit
    def test_format_should_wrap_key_error_with_original_cause(self) -> None:
        """Verify that KeyError during interpolation preserves the original stack trace.

        Given:
            A template string with an undeclared key '{age}' but declared 'name'.
        When:
            The format method is called with only 'name'.
        Then:
            A ConfigError is raised, wrapping the original KeyError as its cause.
        """
        template = StringPromptTemplate(
            template_str="Hello {name}, age {age}", input_variables=["name"]
        )

        with pytest.raises(ConfigError) as exc_info:
            template.format(name="Alice")

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, KeyError)

    @pytest.mark.unit
    def test_format_context_should_return_placeholder_when_nodes_list_is_empty(
        self,
    ) -> None:
        """Verify behavior when providing an empty list of nodes.

        Given:
            An empty list of knowledge nodes.
        When:
            format_context is called.
        Then:
            It returns the specific placeholder string defined in the requirement.
        """
        result = StringPromptTemplate.format_context([])

        assert result == "(No context available)"

    @pytest.mark.rag_edge_case
    @pytest.mark.unit
    def test_format_context_should_handle_complex_unicode_and_normalization(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test robustness against ZWJ emojis and NFC/NFD mixed normalization.

        Given:
            A list of KnowledgeNodes containing Family ZWJ emojis, mixed 'é'
            (NFC/NFD), and ideographic spaces.
        When:
            Formatting these nodes into a single context string.
        Then:
            The context must preserve all original sequences.
        """
        zwj_family = "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466"
        nfc_char = "\u00e9"
        nfd_char = "\u0065\u0301"
        japanese_space = "\u3000"

        nodes: list[KnowledgeNode] = [
            node_factory(content=f"Emoji: {zwj_family}"),
            node_factory(content=f"NFC: {nfc_char}"),
            node_factory(content=f"NFD: {nfd_char}"),
            node_factory(content=f"Space: た{japanese_space}ぼ"),
        ]

        context: str = StringPromptTemplate.format_context(nodes)

        assert zwj_family in context, "ZWJ Emoji sequence was corrupted"
        assert nfc_char in context, "NFC normalization drift detected"
        assert nfd_char in context, "NFD normalization drift detected"
        assert japanese_space in context, "Ideographic space was incorrectly trimmed"
        assert context.count("[") == len(nodes)

    @pytest.mark.unit
    def test_format_context_should_extract_content_from_retrieved_nodes(
        self, retrieved_node_factory: Callable[..., RetrievedNode]
    ) -> None:
        """Test recursive content extraction from RetrievedNode wrappers.

        Given:
            A RetrievedNode wrapping a KnowledgeNode.
        When:
            Formatting context from this node.
        Then:
            The content is extracted and formatted with rank index.
        """
        retrieved = retrieved_node_factory(
            content="Relevant retrieved content.", rank=1
        )

        context: str = StringPromptTemplate.format_context([retrieved])

        assert "Relevant retrieved content." in context
        assert "[1]" in context

    @pytest.mark.integration
    def test_from_file_should_load_template_correctly(self, tmp_path: Path) -> None:
        """Verify loading a template from a temporary file.

        Given:
            A text file containing a valid template string.
        When:
            Initializing StringPromptTemplate via from_file.
        Then:
            The template instance must contain the file content.
        """
        template_content = "Hello {name}, welcome to RAGMark."
        template_file = tmp_path / "greeting.txt"
        template_file.write_text(template_content, encoding="utf-8")

        template = StringPromptTemplate.from_file(
            path=template_file, input_variables=["name"]
        )

        assert template.template_str == template_content
        assert template.format(name="User") == "Hello User, welcome to RAGMark."

    @pytest.mark.rag_edge_case
    @pytest.mark.integration
    def test_from_file_should_raise_error_when_file_missing(
        self, tmp_path: Path
    ) -> None:
        """Verify that FileNotFoundError is raised for non-existent files."""
        missing_file = tmp_path / "ghost.txt"

        with pytest.raises(FileNotFoundError):
            StringPromptTemplate.from_file(missing_file, input_variables=[])

    @pytest.mark.rag_edge_case
    @pytest.mark.unit
    def test_format_should_wrap_value_error_with_original_cause(self) -> None:
        """Verify that ValueError (e.g. format spec mismatch) is wrapped properly.

        Given:
            A template expecting an integer format specifier ':d'.
        When:
            Passing a string value that contradicts the specifier.
        Then:
            A ConfigError is raised, wrapping the original ValueError.
        """
        template = StringPromptTemplate(
            template_str="Age: {age:d}", input_variables=["age"]
        )

        with pytest.raises(ConfigError) as exc_info:
            template.format(age="not_an_int")

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestPromptTemplate:
    """Test suite for PromptTemplate logic (Jinja2 based)."""

    @pytest.mark.unit
    def test_predefined_templates_are_valid(self) -> None:
        """Sanity check for global template constants defined in the module.

        Given:
            Global constants RAG_QA_TEMPLATE and RAG_QA_STRING_TEMPLATE.
        When:
            Inspecting properties and executing a smoke test render.
        Then:
            They must be valid instances and render without runtime errors.
        """
        from ragmark.generation.prompts import RAG_QA_STRING_TEMPLATE, RAG_QA_TEMPLATE

        assert "context_chunks" in RAG_QA_TEMPLATE.input_variables
        assert "user_question" in RAG_QA_TEMPLATE.input_variables

        try:
            result = RAG_QA_TEMPLATE.render(
                context_chunks=["Test Chunk 1", "Test Chunk 2"],
                user_question="Test Question?",
            )
            assert "Test Chunk 1" in result
            assert "Test Question?" in result
        except Exception as e:
            pytest.fail(f"RAG_QA_TEMPLATE failed runtime validation: {e}")

        assert "context" in RAG_QA_STRING_TEMPLATE.input_variables

    @pytest.mark.unit
    def test_render_should_process_jinja2_control_structures(self) -> None:
        """Verify basic Jinja2 control structures rendering.

        Given:
            A template with a loop and conditional logic.
        When:
            Rendering with a list of items.
        Then:
            The output string must reflect the logic correctly.
        """
        template_str = """
        {% for item in items -%}
            - {{ item }}
        {% endfor -%}
        {% if show_footer %}Footer{% endif %}
        """
        template = PromptTemplate(
            template=template_str, input_variables=["items", "show_footer"]
        )

        result = template.render(items=["A", "B"], show_footer=True)

        assert "- A" in result
        assert "- B" in result
        assert "Footer" in result

    @pytest.mark.unit
    def test_init_should_raise_config_error_on_invalid_jinja_syntax(self) -> None:
        """Verify that invalid Jinja2 syntax raises ConfigError during initialization.

        Given:
            A template string with broken syntax (e.g., unclosed tag).
        When:
            Initializing the PromptTemplate.
        Then:
            A ConfigError is raised, wrapping the original TemplateSyntaxError.
        """
        template_str = "Hello {% if name %} {{ name }}"

        with pytest.raises(ConfigError) as exc_info:
            PromptTemplate(template=template_str, input_variables=["name"])
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, TemplateSyntaxError)

    @pytest.mark.unit
    def test_render_should_wrap_runtime_error_with_original_cause(self) -> None:
        """Verify that runtime errors during rendering are wrapped in ConfigError.

        Given:
            A syntactically valid template that causes a runtime error (ZeroDivision).
        When:
            The render method is called.
        Then:
            A ConfigError is raised, wrapping the specific runtime exception.
        """
        template = PromptTemplate(
            template="Calculated: {{ 1 / 0 }}", input_variables=[]
        )

        with pytest.raises(ConfigError) as exc_info:
            template.render()

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    @pytest.mark.unit
    def test_render_should_raise_config_error_on_missing_variable(self) -> None:
        """Verify validation for missing Jinja2 variables.

        Given:
            A template configured with StrictUndefined.
        When:
            Formatting without the required variable.
        Then:
            A ConfigError is raised.
        """
        template = PromptTemplate(template="Hello {{ name }}", input_variables=["name"])

        with pytest.raises(ConfigError) as exc_info:
            template.render()

        assert exc_info.value.__cause__ is None

    @pytest.mark.integration
    def test_from_file_should_load_jinja_template(self, tmp_path: Path) -> None:
        """Verify loading a Jinja2 template from a file.

        Given:
            A file with Jinja2 syntax.
        When:
            Initializing PromptTemplate via from_file.
        Then:
            It renders correctly.
        """
        content = "Hello {{ name|upper }}"
        f = tmp_path / "template.j2"
        f.write_text(content, encoding="utf-8")

        template = PromptTemplate.from_file(f, input_variables=["name"])

        assert template.render(name="alice") == "Hello ALICE"

    @pytest.mark.integration
    def test_from_file_should_raise_error_when_file_missing(
        self, tmp_path: Path
    ) -> None:
        """Verify that FileNotFoundError is raised for non-existent files.

        Given:
            A path to a non-existent file.
        When:
            Initializing PromptTemplate via from_file.
        Then:
            FileNotFoundError is raised directly.
        """
        missing_file = tmp_path / "ghost_template.j2"

        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file(missing_file, input_variables=[])
