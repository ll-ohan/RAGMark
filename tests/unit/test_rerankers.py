"""Unit tests for reranker from_config() pattern.

This module tests the from_config() implementation for rerankers/refiners.
"""

from ragmark.config.profile import RerankerConfig
from ragmark.retrieval.rerankers import CrossEncoderRefiner


class TestCrossEncoderRefinerFromConfig:
    """Tests for CrossEncoderRefiner.from_config()."""

    def test_from_config_basic(self) -> None:
        """Test basic instantiation from config."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            top_k=10,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        assert refiner.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert refiner.device == "cpu"
        assert refiner.top_k == 10

    def test_from_config_with_cuda(self) -> None:
        """Test instantiation with cuda device."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            device="cuda",
            top_k=5,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        assert refiner.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert refiner.device == "cuda"
        assert refiner.top_k == 5

    def test_from_config_custom_top_k(self) -> None:
        """Test instantiation with custom top_k values."""
        for top_k in [3, 5, 10, 20]:
            config = RerankerConfig(
                model_name="cross-encoder/test-model",
                device="cpu",
                top_k=top_k,
            )

            refiner = CrossEncoderRefiner.from_config(config)

            assert refiner.top_k == top_k

    def test_from_config_preserves_all_params(self) -> None:
        """Test that from_config preserves all configuration parameters."""
        config = RerankerConfig(
            model_name="custom-cross-encoder",
            device="mps",
            top_k=15,
        )

        refiner = CrossEncoderRefiner.from_config(config)

        # Verify all attributes match config
        assert refiner.model_name == config.model_name
        assert refiner.device == config.device
        assert refiner.top_k == config.top_k

    def test_from_config_equivalent_to_direct_init(self) -> None:
        """Test that from_config produces same result as direct __init__."""
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        device = "cpu"
        top_k = 10

        # Create via from_config
        config = RerankerConfig(
            model_name=model_name,
            device=device,
            top_k=top_k,
        )
        refiner_from_config = CrossEncoderRefiner.from_config(config)

        # Create via direct init
        refiner_direct = CrossEncoderRefiner(
            model_name=model_name,
            device=device,
            top_k=top_k,
        )

        # Both should have same attributes
        assert refiner_from_config.model_name == refiner_direct.model_name
        assert refiner_from_config.device == refiner_direct.device
        assert refiner_from_config.top_k == refiner_direct.top_k
