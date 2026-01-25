"""Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

from unittest.mock import MagicMock

import pytest

from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc


@pytest.fixture
def sample_source_doc() -> SourceDoc:
    """Create a sample SourceDoc for testing.

    Returns:
        A valid SourceDoc instance.
    """
    return SourceDoc(
        content="This is a sample document content for testing purposes.",
        metadata={"title": "Test Document", "author": "Test Author"},
        source_id="test-source-123",
        mime_type="text/plain",
        page_count=1,
    )


@pytest.fixture
def sample_knowledge_node() -> KnowledgeNode:
    """Create a sample KnowledgeNode for testing.

    Returns:
        A valid KnowledgeNode instance.
    """
    return KnowledgeNode(
        node_id="node-123",
        content="This is a sample chunk of text.",
        source_id="test-source-123",
        metadata={"source.title": "Test Document"},
        position=NodePosition(
            start_char=0,
            end_char=32,
            page=1,
            section="Introduction",
        ),
        dense_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        sparse_vector={1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8},
    )


@pytest.fixture
def mock_ingestor() -> MagicMock:
    """Create a mock BaseIngestor.

    Returns:
        MagicMock configured to behave like a BaseIngestor.
    """
    from ragmark.forge.ingestors import BaseIngestor

    ingestor = MagicMock(spec=BaseIngestor)
    ingestor.supported_formats = {".txt", ".md"}
    return ingestor


@pytest.fixture
def mock_fragmenter() -> MagicMock:
    """Create a mock BaseFragmenter.

    Returns:
        MagicMock configured to behave like a BaseFragmenter.
    """
    from ragmark.forge.fragmenters import BaseFragmenter

    fragmenter = MagicMock(spec=BaseFragmenter)
    fragmenter.chunk_size = 256
    fragmenter.overlap = 64
    return fragmenter


@pytest.fixture
def sample_yaml_config() -> str:
    """Sample YAML configuration for testing.

    Returns:
        Valid YAML configuration string.
    """
    return """
ingestor:
  backend: fitz
  options: {}

fragmenter:
  strategy: token
  chunk_size: 512
  overlap: 128
  options:
    tokenizer: cl100k_base

embedder:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

index:
  backend: memory
  collection_name: test_collection
  embedding_dim: 384
  connection: null

retrieval:
  mode: dense
  top_k: 10
  alpha: null
  reranker: null

generator:
  model_path: ./models/test-model.gguf
  context_window: 4096
  max_new_tokens: 512
  temperature: 0.7

evaluation:
  metrics:
    - recall@k
    - mrr
  trial_cases_path: ./data/test_cases.jsonl
"""
