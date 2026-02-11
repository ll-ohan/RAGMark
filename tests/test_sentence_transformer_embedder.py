"""Unit tests for the SentenceTransformerEmbedder wrapper."""

import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ragmark.config.profile import EmbedderConfig
from ragmark.exceptions import EmbeddingError, UnsupportedBackendError
from ragmark.index.embedders import SentenceTransformerEmbedder


@pytest.mark.unit
class TestSentenceTransformerEmbedder:
    """Test suite for SentenceTransformerEmbedder isolating wrapper logic."""

    def test_init_should_configure_defaults_and_lazy_loading_when_only_model_name_provided(
        self,
    ) -> None:
        """
        Given: A model name string.
        When: The embedder is initialized without optional parameters.
        Then: Defaults (cpu, batch=32) should be set and model should remain unloaded (None).
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        assert embedder.model_name == "test-model"
        assert embedder.device == "cpu"
        assert embedder.batch_size == 32
        assert embedder._model is None, "Model should be lazy loaded"  # type: ignore
        assert embedder._dim is None  # type: ignore

    def test_init_should_respect_custom_parameters_when_provided(self) -> None:
        """
        Given: Custom parameters for device and batch size.
        When: The embedder is initialized.
        Then: The instance attributes should reflect these custom values.
        """
        embedder = SentenceTransformerEmbedder(
            model_name="custom-model",
            device="cuda",
            batch_size=64,
        )

        assert embedder.model_name == "custom-model"
        assert embedder.device == "cuda"
        assert embedder.batch_size == 64

    def test_from_config_should_create_instance_matching_configuration(
        self, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """
        Given: A valid EmbedderConfig object.
        When: The embedder is created via from_config factory.
        Then: The created instance should match the config attributes.
        """
        config = embedder_config_factory(
            model_name="config-model", device="cuda:1", batch_size=128
        )

        embedder = SentenceTransformerEmbedder.from_config(config)

        assert embedder.model_name == "config-model"
        assert embedder.device == "cuda:1"
        assert embedder.batch_size == 128

    def test_load_model_should_load_backend_on_first_use(self) -> None:
        """
        Given: An uninitialized embedder.
        When: _load_model is called (explicitly or implicitly).
        Then: The sentence-transformers model should be loaded and dimensions cached.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")
        assert embedder._model is None  # type: ignore

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            embedder._load_model()  # type: ignore

        assert embedder._model is mock_model  # type: ignore
        assert embedder._dim == 384  # type: ignore
        mock_st.SentenceTransformer.assert_called_once_with("test-model", device="cpu")

    def test_load_model_should_cache_instance_to_prevent_multiple_initializations(
        self,
    ) -> None:
        """
        Given: An embedder that has already loaded its model.
        When: _load_model is called a second time.
        Then: The same model instance should be returned without calling the backend again.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            embedder._load_model()  # type: ignore
            first_model = embedder._model  # type: ignore

            embedder._load_model()  # type: ignore
            second_model = embedder._model  # type: ignore

        assert first_model is second_model
        assert mock_st.SentenceTransformer.call_count == 1

    def test_load_model_should_raise_unsupported_backend_error_when_dependency_missing(
        self,
    ) -> None:
        """
        Given: The sentence-transformers library is not installed.
        When: Model loading is triggered.
        Then: UnsupportedBackendError should be raised with a clear message.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(UnsupportedBackendError, match="sentence-transformers"):
                embedder._load_model()  # type: ignore

    def test_load_model_should_raise_embedding_error_preserving_cause_when_loading_fails(
        self,
    ) -> None:
        """
        Given: The backend raises a RuntimeError during initialization.
        When: Model loading is triggered.
        Then: EmbeddingError should be raised, and the original exception should be preserved in __cause__.
        """
        embedder = SentenceTransformerEmbedder(model_name="invalid-model")

        mock_st = MagicMock()
        original_error = RuntimeError("Model file not found")
        mock_st.SentenceTransformer.side_effect = original_error

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            with pytest.raises(
                EmbeddingError, match="Failed to load model"
            ) as exc_info:
                embedder._load_model()  # type: ignore

            assert exc_info.value.__cause__ is original_error

    def test_embed_should_return_list_of_vectors_when_input_is_valid(self) -> None:
        """
        Given: A list of text strings.
        When: embed is called.
        Then: It should return a list of list of floats matching the expected dimensions.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = embedder.embed(["text 1", "text 2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [1.0, 0.0, 0.0]

        call_args = mock_model.encode.call_args
        assert call_args[1]["convert_to_numpy"] is True
        assert call_args[1]["show_progress_bar"] is False

    def test_embed_should_return_empty_list_when_input_is_empty(self) -> None:
        """
        Given: An empty list of texts.
        When: embed is called.
        Then: It should return an empty list immediately without calling the model.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = embedder.embed([])

        assert result == []

    def test_embed_should_pass_configured_batch_size_to_model_encode(self) -> None:
        """
        Given: An embedder configured with a specific batch size.
        When: embed is called.
        Then: The batch_size parameter should be passed correctly to the backend encode method.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model", batch_size=128)

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            embedder.embed(["text"])

        call_args = mock_model.encode.call_args
        assert call_args[1]["batch_size"] == 128

    def test_embed_should_wrap_runtime_errors_in_embedding_error_preserving_stack_trace(
        self,
    ) -> None:
        """
        Given: The backend model raises an exception during encoding.
        When: embed is called.
        Then: EmbeddingError should be raised with the original exception as cause.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        original_error = RuntimeError("CUDA OOM")
        mock_model.encode.side_effect = original_error
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            with pytest.raises(
                EmbeddingError, match="Embedding computation failed"
            ) as exc_info:
                embedder.embed(["text"])

            assert exc_info.value.__cause__ is original_error

    def test_embedding_dim_should_trigger_model_loading_when_accessed(self) -> None:
        """
        Given: An uninitialized embedder.
        When: embedding_dim property is accessed.
        Then: The model should be loaded to retrieve the dimension.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")
        assert embedder._model is None  # type: ignore

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            dim = embedder.embedding_dim

        assert dim == 768
        assert embedder._model is not None  # type: ignore

    def test_supports_sparse_should_return_false(self) -> None:
        """
        Given: A SentenceTransformerEmbedder instance.
        When: supports_sparse is checked.
        Then: It should always return False as this is a dense-only embedder.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")
        assert embedder.supports_sparse is False

    def test_embed_should_handle_unicode_and_special_characters_correctly(self) -> None:
        """
        Given: Texts containing complex unicode, NFD normalization, and ideographic spaces.
        When: embed is called.
        Then: The embedder should accept them and return vectors without crashing.

        Note: We are mocking the backend, so we verify that the wrapper passes these strings
        intact to the backend's encode method.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        complex_text = "NFD: e" + "\u0301" + " vs NFC: \u00e9"
        emoji_text = "Family: 👨‍👩‍👧‍👦"
        ideographic_space = "Start\u3000End"

        texts = [complex_text, emoji_text, ideographic_space]

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 3))
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = embedder.embed(texts)

        assert len(result) == 3

        called_texts = mock_model.encode.call_args[0][0]
        assert called_texts == texts

    def test_embed_should_convert_numpy_arrays_to_python_lists(self) -> None:
        """
        Given: The backend returns numpy arrays.
        When: embed is called.
        Then: The result should be converted to standard Python lists for JSON serialization compatibility.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_embeddings = np.array([[1.5, 2.5]], dtype=np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = embedder.embed(["text"])

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)
        assert abs(result[0][0] - 1.5) < 1e-6

    def test_embed_should_disable_normalization_by_default(self) -> None:
        """
        Given: Default initialization.
        When: embed is called.
        Then: normalize_embeddings should be passed as False to the backend.
        """
        embedder = SentenceTransformerEmbedder(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0]])
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            embedder.embed(["text"])

        assert mock_model.encode.call_args[1]["normalize_embeddings"] is False


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.asyncio
class TestEmbedderRateLimiting:
    """Tests for SentenceTransformerEmbedder rate limiting integration."""

    async def test_embedder_with_rate_limit_should_throttle_requests(self) -> None:
        """Verifies embedder respects rate_limit parameter.

        Given:
            SentenceTransformerEmbedder with rate_limit=2.0 and batch_size=2.
        When:
            Embedding 10 texts via embed_async().
        Then:
            Operation takes approximately 1.5 seconds (5 batches, 2/s after burst).
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=2,
            rate_limit=2.0,
        )

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        texts = [f"Test text {i}" for i in range(10)]

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # 5 batches at 2/s = ~1.5s rate limiting (2 immediate + 3 at 2/s)
        assert 1.0 < duration < 6.0, f"Rate limit not enforced: {duration:.2f}s"

    async def test_embedder_without_rate_limit_should_run_unrestricted(self) -> None:
        """Verifies embedder without rate_limit runs at full speed.

        Given:
            SentenceTransformerEmbedder with rate_limit=None.
        When:
            Embedding 10 texts (after model warmup).
        Then:
            Completes quickly without artificial delays.
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            rate_limit=None,
        )

        texts = [f"Quick test {i}" for i in range(10)]

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # Should complete in <2s without rate limiting (model already loaded)
        assert duration < 2.0, f"Unrestricted took too long: {duration:.2f}s"

    async def test_embedder_rate_limit_should_work_with_batching(self) -> None:
        """Verifies rate limiting applies per batch, not per text.

        Given:
            SentenceTransformerEmbedder with rate_limit=2.0, batch_size=5.
        When:
            Embedding 15 texts (processed as 3 batches).
        Then:
            Duration reflects 3 batch acquisitions at 2/s (~0.5 second).
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=5,
            rate_limit=2.0,
        )

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        texts = [f"Batch test {i}" for i in range(15)]

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # 3 batches at 2/s = ~0.5s rate limiting
        assert 0.3 < duration < 6.0, f"Batch rate limiting: {duration:.2f}s"
