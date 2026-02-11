"""Test suite for llama.cpp driver with mocked model loading.

Validates LlamaCppDriver behavior without requiring actual GGUF model files by
mocking the llama-cpp-python library for initialization, generation, and streaming.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ragmark.exceptions import GenerationError
from ragmark.generation.drivers import LlamaCppDriver
from ragmark.schemas.generation import GenerationResult


@pytest.mark.unit
class TestLlamaCppDriverInitialization:
    """Test suite for LlamaCppDriver initialization and lazy loading."""

    def test_init_should_store_configuration_without_loading_model(
        self, tmp_path: Path
    ):
        """Verify model is not loaded during __init__.

        Given:
            Valid model path and configuration parameters.
        When:
            Initializing LlamaCppDriver.
        Then:
            Configuration is stored but model remains None (lazy loading).
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        driver = LlamaCppDriver(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
        )

        assert driver.model_path == model_path
        assert driver._n_ctx == 2048  # type: ignore
        assert driver._n_gpu_layers == 0  # type: ignore
        assert driver._model is None  # type: ignore
        assert driver._executor is None  # type: ignore

    @pytest.mark.asyncio
    async def test_aenter_should_load_model_lazily(self, tmp_path: Path):
        """Verify model loading occurs in __aenter__.

        Given:
            Unloaded LlamaCppDriver instance.
        When:
            Entering async context manager.
        Then:
            Model is loaded via thread pool executor.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.n_ctx.return_value = 2048

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path, n_ctx=2048)

            async with driver:
                assert driver._model is not None  # type: ignore
                assert driver._executor is not None  # type: ignore

    @pytest.mark.asyncio
    async def test_aenter_should_raise_error_when_model_file_missing(
        self, tmp_path: Path
    ):
        """Verify error handling for missing model file.

        Given:
            Path to non-existent model file.
        When:
            Entering async context manager.
        Then:
            GenerationError is raised with clear message.
        """
        missing_path = tmp_path / "nonexistent.gguf"

        driver = LlamaCppDriver(model_path=missing_path)

        with pytest.raises(GenerationError, match="Failed to load model"):
            async with driver:
                pass

    @pytest.mark.asyncio
    async def test_aenter_should_raise_error_when_llama_cpp_not_installed(
        self, tmp_path: Path
    ):
        """Verify error handling when llama-cpp-python not installed.

        Given:
            System without llama-cpp-python package.
        When:
            Attempting to load model.
        Then:
            GenerationError with installation instructions is raised.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        driver = LlamaCppDriver(model_path=model_path)

        original_import = __import__

        def mock_import(name: str, *args: Any, **kwargs: Any):
            if name == "llama_cpp":
                raise ImportError("No module named 'llama_cpp'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(GenerationError, match="Failed to load model"):
                async with driver:
                    pass

    @pytest.mark.asyncio
    async def test_aexit_should_cleanup_executor(self, tmp_path: Path):
        """Verify thread pool executor cleanup on context exit.

        Given:
            Loaded LlamaCppDriver with active executor.
        When:
            Exiting async context manager.
        Then:
            Thread pool executor is shut down properly.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                executor = driver._executor  # type: ignore
                assert executor is not None

            assert executor._shutdown is True


@pytest.mark.unit
class TestLlamaCppDriverGeneration:
    """Test suite for text generation methods."""

    @pytest.mark.asyncio
    async def test_generate_should_return_generation_result(self, tmp_path: Path):
        """Verify successful text generation.

        Given:
            Loaded driver and valid prompt.
        When:
            Calling generate method.
        Then:
            Returns GenerationResult with text and usage stats.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [
                {
                    "text": "Generated response text.",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                result = await driver.generate(
                    prompt="Test prompt",
                    max_tokens=100,
                    temperature=0.7,
                )

                assert isinstance(result, GenerationResult)
                assert result.text == "Generated response text."
                assert result.usage.prompt_tokens == 10
                assert result.usage.completion_tokens == 5
                assert result.usage.total_tokens == 15
                assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_should_raise_error_when_model_not_loaded(self):
        """Verify error when generating without loaded model.

        Given:
            Driver instance outside context manager.
        When:
            Calling generate method.
        Then:
            GenerationError is raised indicating model not loaded.
        """
        driver = LlamaCppDriver(model_path="dummy.gguf")

        with pytest.raises(GenerationError, match="Model not loaded"):
            await driver.generate(prompt="Test", max_tokens=10)

    @pytest.mark.asyncio
    async def test_generate_should_handle_generation_failure(self, tmp_path: Path):
        """Verify error handling during generation failure.

        Given:
            Loaded driver where llama.cpp raises exception.
        When:
            Calling generate method.
        Then:
            GenerationError is raised with proper cause chaining.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.side_effect = RuntimeError("Model inference failed")

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                with pytest.raises(GenerationError, match="Text generation failed"):
                    await driver.generate(prompt="Test", max_tokens=10)

    @pytest.mark.asyncio
    async def test_generate_stream_should_yield_text_chunks(self, tmp_path: Path):
        """Verify streaming generation yields chunks.

        Given:
            Loaded driver with streaming enabled.
        When:
            Calling generate_stream method.
        Then:
            Yields text chunks as they become available.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_stream = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]},
        ]

        mock_llama = MagicMock()
        mock_llama.return_value = iter(mock_stream)

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                chunks: list[str] = []
                async for chunk in driver.generate_stream(
                    prompt="Test",
                    max_tokens=10,
                ):
                    chunks.append(chunk)

                assert chunks == ["Hello", " world", "!"]


@pytest.mark.unit
class TestLlamaCppDriverTokenCounting:
    """Test suite for token counting functionality."""

    @pytest.mark.asyncio
    async def test_count_tokens_should_return_token_count(self, tmp_path: Path):
        """Verify token counting accuracy.

        Given:
            Loaded driver and text string.
        When:
            Calling count_tokens method.
        Then:
            Returns accurate token count from model tokenizer.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.tokenize.return_value = [1, 2, 3, 4, 5]

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                count = driver.count_tokens("Hello world")

                assert count == 5
                mock_llama.tokenize.assert_called_once_with(b"Hello world")

    def test_count_tokens_should_raise_error_when_model_not_loaded(self):
        """Verify error when counting tokens without loaded model.

        Given:
            Driver instance outside context manager.
        When:
            Calling count_tokens method.
        Then:
            GenerationError is raised.
        """
        driver = LlamaCppDriver(model_path="dummy.gguf")

        with pytest.raises(GenerationError, match="Model not loaded"):
            driver.count_tokens("Test text")

    def test_context_window_property_should_return_configured_value(self):
        """Verify context_window property returns configured n_ctx.

        Given:
            Driver initialized with specific n_ctx value.
        When:
            Accessing context_window property.
        Then:
            Returns the configured context window size.
        """
        driver = LlamaCppDriver(model_path="dummy.gguf", n_ctx=4096)

        assert driver.context_window == 4096


@pytest.mark.unit
class TestLlamaCppDriverEdgeCases:
    """Test suite for edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_generate_with_empty_stop_sequences(self, tmp_path: Path):
        """Verify generation handles empty stop sequences list.

        Given:
            Driver with stop=[] parameter.
        When:
            Generating text.
        Then:
            Generation proceeds without error.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                result = await driver.generate(
                    prompt="Test",
                    max_tokens=10,
                    stop=[],
                )

                assert result.text == "Response"

    @pytest.mark.asyncio
    async def test_generate_with_none_stop_sequences(self, tmp_path: Path):
        """Verify generation handles None stop parameter.

        Given:
            Driver with stop=None parameter.
        When:
            Generating text.
        Then:
            stop is converted to empty list internally.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_text("fake model")

        mock_llama = MagicMock()
        mock_llama.return_value = {
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        with patch("llama_cpp.Llama", return_value=mock_llama):
            driver = LlamaCppDriver(model_path=model_path)

            async with driver:
                result = await driver.generate(
                    prompt="Test",
                    max_tokens=10,
                    stop=None,
                )

                assert result.text == "Response"
                mock_llama.assert_called_with(
                    "Test",
                    max_tokens=10,
                    temperature=0.7,
                    stop=[],
                    echo=False,
                )
