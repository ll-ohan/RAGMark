"""Unit tests for the reranking module."""

from collections.abc import Callable

import pytest

from ragmark.config.profile import RerankerConfig
from ragmark.retrieval.rerankers import CrossEncoderRefiner
from ragmark.schemas.retrieval import RetrievedNode


@pytest.mark.unit
class TestCrossEncoderRefiner:
    """Tests for CrossEncoderRefiner instantiation and behavior."""

    def test_from_config_should_instantiate_refiner_with_provided_parameters(
        self,
    ) -> None:
        """Verifies that the refiner is correctly initialized from a configuration object.

        Given:
            A RerankerConfig object with specific model, device, and top_k settings.
        When:
            CrossEncoderRefiner.from_config is called with this configuration.
        Then:
            The created instance should have attributes matching the configuration.
        """
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            top_k=10,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        assert refiner.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert refiner.device == "cpu"
        assert refiner.top_k == 10

    def test_from_config_should_respect_cuda_device_setting(self) -> None:
        """Verifies that the refiner accepts 'cuda' as a device parameter.

        Given:
            A RerankerConfig object with device set to 'cuda'.
        When:
            CrossEncoderRefiner.from_config is instantiated.
        Then:
            The instance device attribute is 'cuda'.
        """
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cuda",
            top_k=5,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        assert refiner.device == "cuda"

    @pytest.mark.parametrize("top_k", [3, 5, 10, 20])
    def test_from_config_should_apply_custom_top_k_values(self, top_k: int) -> None:
        """Verifies that various top_k values are correctly applied.

        Given:
            A RerankerConfig with a specific top_k value.
        When:
            CrossEncoderRefiner is instantiated via from_config.
        Then:
            The instance's top_k attribute matches the config's top_k.
        """
        config = RerankerConfig(
            model_name="cross-encoder/test-model",
            device="cpu",
            top_k=top_k,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        assert refiner.top_k == top_k

    def test_refine_should_raise_not_implemented_error(
        self, retrieved_node_factory: Callable[..., RetrievedNode]
    ) -> None:
        """Verifies that the refine method raises NotImplementedError with correct message.

        Given:
            An instantiated CrossEncoderRefiner and a list of candidate nodes.
        When:
            The refine method is called with valid arguments (candidates, top_k).
        Then:
            A NotImplementedError should be raised with a specific Phase 2 message.
        """
        config = RerankerConfig(
            model_name="dummy-model",
            device="cpu",
            top_k=5,
        )
        refiner = CrossEncoderRefiner.from_config(config)
        candidates = [retrieved_node_factory() for _ in range(10)]

        query = "test query"

        with pytest.raises(
            NotImplementedError,
            match="CrossEncoderRefiner will be implemented in Phase 2",
        ):
            refiner.refine(query=query, candidates=candidates, top_k=5)
