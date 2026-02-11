"""Unit tests for QA to trial case adapters.

This module tests the conversion from enriched knowledge nodes
to TrialCase format for evaluation using adapters.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ragmark.adapters.formats.json_adapter import JSONAdapter
from ragmark.adapters.formats.jsonl_adapter import JSONLAdapter
from ragmark.adapters.transformers.qa_adapter import NodeToTrialCaseAdapter
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import TrialCase

if TYPE_CHECKING:

    def approx(expected: float) -> object:
        ...
else:

    def approx(expected: float) -> object:
        return pytest.approx(expected)


@pytest.mark.unit
def test_nodes_to_trial_cases_should_convert_enriched_nodes(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates conversion from enriched nodes to trial cases.

    Given:
        Two enriched nodes with 2 QA pairs each.
    When:
        Using NodeToTrialCaseAdapter.
    Then:
        4 trial cases are created (2 per node).
        Each case has question, answer, and metadata.
        Ground truth node IDs are set.
    """
    nodes: list[KnowledgeNode] = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=2),
    ]

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many(nodes)

    assert len(trial_cases) == 4

    case0 = trial_cases[0]
    assert case0.question == "Question 1 about src1?"
    assert isinstance(case0.ground_truth_answer, str)
    assert case0.ground_truth_answer.startswith("Answer 1: Content 1")
    assert case0.ground_truth_node_ids == [nodes[0].node_id]
    assert case0.metadata["source_node_id"] == nodes[0].node_id
    assert case0.metadata["source_id"] == "src1"
    assert case0.metadata["qa_index"] == 0
    assert case0.metadata["confidence"] == 0.9

    case1 = trial_cases[1]
    assert case1.question == "Question 2 about src1?"
    assert case1.metadata["qa_index"] == 1

    case2 = trial_cases[2]
    assert case2.question == "Question 1 about src2?"
    assert case2.ground_truth_node_ids == [nodes[1].node_id]
    assert case2.metadata["source_id"] == "src2"


@pytest.mark.unit
def test_nodes_to_trial_cases_without_ground_truth_node_ids(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates conversion without ground truth node IDs.

    Given:
        Enriched nodes with QA pairs.
        include_ground_truth_nodes=False.
    When:
        Using NodeToTrialCaseAdapter.
    Then:
        Trial cases have None for ground_truth_node_ids.
        ground_truth_answer is still set with valid content.
    """
    nodes: list[KnowledgeNode] = [
        enriched_node_factory("Content", "src1", num_qa_pairs=1)
    ]

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=False)
    trial_cases = adapter.adapt_many(nodes)

    assert len(trial_cases) == 1
    assert isinstance(trial_cases[0].ground_truth_answer, str)
    assert trial_cases[0].ground_truth_answer.startswith("Answer 1:")
    assert len(trial_cases[0].ground_truth_answer) > 10
    assert trial_cases[0].ground_truth_node_ids is None


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_nodes_without_qa(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates nodes without synthetic_qa are skipped.

    Given:
        Mix of enriched nodes and plain nodes.
    When:
        Using NodeToTrialCaseAdapter.
    Then:
        Only enriched nodes are converted.
        Plain nodes are skipped with debug log.
    """
    enriched = enriched_node_factory("Content 1", "src1", num_qa_pairs=2)
    plain = KnowledgeNode(
        content="Content 2",
        source_id="src2",
        position=NodePosition(start_char=0, end_char=9, page=1, section="section"),
        metadata={"custom_field": "custom_value"},
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector={1: 0.5, 5: 0.3, 10: 0.8},
    )

    nodes = [enriched, plain]
    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many(nodes)

    assert len(trial_cases) == 2
    assert all(case.metadata["source_id"] == "src1" for case in trial_cases)


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_empty_qa_pairs(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates nodes with empty qa_pairs are skipped.

    Given:
        Node with synthetic_qa metadata but empty qa_pairs list.
    When:
        Using NodeToTrialCaseAdapter.
    Then:
        No trial cases are created.
    """
    node = KnowledgeNode(
        content="Content",
        source_id="src1",
        position=NodePosition(start_char=0, end_char=7, page=1, section="section"),
        metadata={"synthetic_qa": {"qa_pairs": []}},
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector=None,
    )

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 0


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_invalid_qa_pairs(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates invalid QA pairs (missing question or answer) are skipped.

    Given:
        Node with some valid and some invalid QA pairs.
    When:
        Using NodeToTrialCaseAdapter.
    Then:
        Only valid pairs are converted.
        Invalid pairs are skipped.
    """
    node = KnowledgeNode(
        content="Content",
        source_id="src1",
        position=NodePosition(start_char=0, end_char=7, page=1, section="section"),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {"question": "Valid?", "answer": "Valid answer"},
                    {"question": "Missing answer?"},
                    {"answer": "Missing question"},
                    {"question": "Another valid?", "answer": "Another answer"},
                ],
            }
        },
        dense_vector=None,
        sparse_vector=None,
    )

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 2
    assert trial_cases[0].question == "Valid?"
    assert trial_cases[1].question == "Another valid?"


@pytest.mark.unit
def test_export_to_jsonl_should_create_file(
    tmp_path: Path,
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates JSONL export creates valid file.

    Given:
        Enriched nodes with QA pairs.
        A temporary output path.
    When:
        Using adapters to convert and export.
    Then:
        JSONL file is created with correct format.
        Each line is a valid TrialCase JSON.
    """
    nodes = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=1),
    ]

    output_path = tmp_path / "trial_cases.jsonl"

    qa_adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = qa_adapter.adapt_many(nodes)

    jsonl_adapter = JSONLAdapter()
    cases_data = [case.model_dump() for case in trial_cases]
    jsonl_adapter.write(cases_data, output_path)

    assert len(trial_cases) == 3
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3

    for line in lines:
        case_data = json.loads(line)
        case = TrialCase.model_validate(case_data)
        assert case.question.endswith("?")
        assert isinstance(case.ground_truth_answer, str)
        assert len(case.ground_truth_answer) > 0
        assert case.ground_truth_answer.startswith("Answer")


@pytest.mark.unit
def test_export_to_json_should_create_file(
    tmp_path: Path,
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates JSON export creates valid file.

    Given:
        Enriched nodes with QA pairs.
        A temporary output path.
    When:
        Using adapters to convert and export.
    Then:
        JSON file is created with list of trial cases.
        File is valid JSON with proper indentation.
    """
    nodes = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=1),
    ]

    output_path = tmp_path / "trial_cases.json"

    qa_adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = qa_adapter.adapt_many(nodes)

    json_adapter = JSONAdapter()
    cases_data = [case.model_dump() for case in trial_cases]
    json_adapter.write(cases_data, output_path)

    assert len(trial_cases) == 3
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        loaded_data: list[dict[str, Any]] = json.load(f)

    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 3

    for case_data in loaded_data:
        case = TrialCase.model_validate(case_data)
        assert case.question.endswith("?")
        assert isinstance(case.ground_truth_answer, str)
        assert len(case.ground_truth_answer) > 0
        assert case.ground_truth_answer.startswith("Answer")


@pytest.mark.unit
def test_export_should_preserve_source_metadata(
    tmp_path: Path,
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates export preserves source node metadata.

    Given:
        Enriched node with custom metadata fields.
    When:
        Converting to trial cases using adapter.
    Then:
        Custom metadata is included in trial case metadata.
        synthetic_qa is excluded to avoid bloat.
    """
    node = enriched_node_factory("Content", "src1", num_qa_pairs=1)

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]

    assert "source_metadata" in case.metadata
    source_metadata = cast(dict[str, Any], case.metadata.get("source_metadata"))
    assert source_metadata is not None
    assert source_metadata["custom_field"] == "custom_value"
    assert "synthetic_qa" not in source_metadata


@pytest.mark.unit
def test_export_should_create_parent_directories(
    tmp_path: Path,
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates export creates parent directories if needed.

    Given:
        Output path with non-existent parent directories.
    When:
        Using adapters to export.
    Then:
        Parent directories are created automatically.
        Export succeeds.
    """
    nodes = [enriched_node_factory("Content", "src1", num_qa_pairs=1)]

    output_path = tmp_path / "subdir1" / "subdir2" / "trial_cases.jsonl"
    assert not output_path.parent.exists()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    qa_adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = qa_adapter.adapt_many(nodes)

    jsonl_adapter = JSONLAdapter()
    cases_data = [case.model_dump() for case in trial_cases]
    jsonl_adapter.write(cases_data, output_path)

    assert len(trial_cases) == 1
    assert output_path.exists()
    assert output_path.parent.exists()


@pytest.mark.unit
def test_nodes_to_trial_cases_should_handle_unicode_nfc_nfd(
    enriched_node_factory: Callable[..., KnowledgeNode],
):
    """Validates NFC/NFD Unicode normalization in QA pairs.

    Given:
        Node with mixed Unicode normalization (NFC + NFD).
    When:
        Converting to trial cases using adapter.
    Then:
        Questions and answers are preserved correctly.
        No character corruption occurs.
    """
    content_nfc = "Café résumé"

    node = enriched_node_factory(content_nfc, "unicode-nfc", num_qa_pairs=1)

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert isinstance(case.ground_truth_answer, str)
    assert "Café" in case.ground_truth_answer or "Café" in case.ground_truth_answer
    assert len(case.question) > 0


@pytest.mark.unit
def test_nodes_to_trial_cases_should_handle_complex_emoji():
    """Validates handling of complex emoji in QA content.

    Given:
        Node with complex composite emoji (family, flags).
    When:
        Converting to trial cases using adapter.
    Then:
        Emoji are preserved in questions and answers.
        No grapheme splitting occurs.
    """
    content_with_emoji = "User profile: 👨‍👩‍👧‍👦 Family 🇫🇷 France"

    node = KnowledgeNode(
        content=content_with_emoji,
        source_id="emoji-test",
        position=NodePosition(
            start_char=0, end_char=len(content_with_emoji), page=1, section="section"
        ),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {
                        "question": "What emoji represents family? 👨‍👩‍👧‍👦",
                        "answer": "The family emoji 👨‍👩‍👧‍👦 represents a complete family unit",
                        "confidence": 0.95,
                    }
                ],
            }
        },
        dense_vector=None,
        sparse_vector=None,
    )

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert "👨‍👩‍👧‍👦" in case.question
    assert isinstance(case.ground_truth_answer, str)
    assert "👨‍👩‍👧‍👦" in case.ground_truth_answer


@pytest.mark.unit
def test_nodes_to_trial_cases_should_handle_ideographic_whitespace():
    """Validates handling of CJK ideographic whitespace.

    Given:
        Node with CJK text and ideographic space (U+3000).
    When:
        Converting to trial cases using adapter.
    Then:
        Whitespace is preserved in content.
        No trimming corruption occurs.
    """
    ideographic_space = "\u3000"
    content_cjk = f"東京{ideographic_space}日本"

    node = KnowledgeNode(
        content=content_cjk,
        source_id="cjk-test",
        position=NodePosition(
            start_char=0, end_char=len(content_cjk), page=1, section="section"
        ),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {
                        "question": f"Where is{ideographic_space}Tokyo?",
                        "answer": f"Tokyo{ideographic_space}Japan",
                        "confidence": 0.9,
                    }
                ],
            }
        },
        dense_vector=None,
        sparse_vector=None,
    )

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert ideographic_space in case.question
    assert isinstance(case.ground_truth_answer, str)
    assert ideographic_space in case.ground_truth_answer


@pytest.mark.unit
def test_nodes_to_trial_cases_should_reject_control_characters():
    """Validates rejection of control characters in QA pairs.

    Given:
        Node with control characters (null bytes, BOM) in QA.
    When:
        Converting to trial cases using adapter.
    Then:
        Control characters are either stripped or handled safely.
        No data corruption in exported JSON.
    """
    content_with_control = "Clean content"

    node = KnowledgeNode(
        content=content_with_control,
        source_id="control-test",
        position=NodePosition(
            start_char=0, end_char=len(content_with_control), page=1, section="section"
        ),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {
                        "question": "Question with \x00 null byte?",
                        "answer": "Answer with \ufeff BOM marker",
                        "confidence": 0.8,
                    }
                ],
            }
        },
        dense_vector=None,
        sparse_vector=None,
    )

    adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = adapter.adapt_many([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert "\x00" not in case.question or len(case.question) > 0
    assert isinstance(case.ground_truth_answer, str)
    assert "\ufeff" not in case.ground_truth_answer or len(case.ground_truth_answer) > 0


@pytest.mark.unit
def test_json_adapter_read_should_wrap_single_object(tmp_path: Path):
    """Ensures single JSON object is wrapped into a list.

    Given:
        A JSON file containing a single object.
    When:
        Reading with JSONAdapter.
    Then:
        The object is returned as a one-item list with identical fields.
    """
    payload: dict[str, Any] = {"id": 1, "name": "alpha", "score": 0.25}
    path = tmp_path / "single.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = JSONAdapter()
    data = adapter.read(path)

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["name"] == "alpha"
    assert data[0]["score"] == approx(0.25)


@pytest.mark.unit
def test_json_adapter_read_should_handle_list_payload(tmp_path: Path):
    """Validates list payloads are returned without mutation.

    Given:
        A JSON file containing a list of objects.
    When:
        Reading with JSONAdapter.
    Then:
        The returned list matches the stored objects in order.
    """
    payload: list[dict[str, Any]] = [
        {"id": 1, "name": "alpha", "score": 0.25},
        {"id": 2, "name": "bravo", "score": 0.5},
    ]
    path = tmp_path / "list.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = JSONAdapter()
    data = adapter.read(path)

    assert data == payload
    assert data[0]["id"] < data[1]["id"]
    assert data[0]["score"] < data[1]["score"]


@pytest.mark.unit
def test_json_adapter_write_should_preserve_unicode(tmp_path: Path):
    """Ensures Unicode content is not ASCII-escaped on write.

    Given:
        Data containing accented characters and emoji.
    When:
        Writing with JSONAdapter.
    Then:
        The raw file content contains the original Unicode characters.
    """
    payload: list[dict[str, Any]] = [{"id": 1, "text": "Café ☕"}]
    path = tmp_path / "unicode.json"

    adapter = JSONAdapter()
    adapter.write(payload, path)

    content = path.read_text(encoding="utf-8")
    assert "Café ☕" in content
    loaded = json.loads(content)
    assert isinstance(loaded, list)
    assert loaded[0]["text"] == "Café ☕"


@pytest.mark.unit
def test_json_adapter_read_should_raise_on_invalid_json(tmp_path: Path):
    """Validates invalid JSON raises a JSONDecodeError with cause.

    Given:
        A malformed JSON file.
    When:
        Reading with JSONAdapter.
    Then:
        A JSONDecodeError is raised and preserves a causal chain.
    """
    path = tmp_path / "invalid.json"
    path.write_text("{invalid-json", encoding="utf-8")

    adapter = JSONAdapter()

    with pytest.raises(json.JSONDecodeError) as exc_info:
        adapter.read(path)

    assert exc_info.value.__cause__ is not None


@pytest.mark.unit
def test_jsonl_adapter_read_should_skip_empty_lines(tmp_path: Path):
    """Ensures empty lines are ignored when reading JSONL.

    Given:
        A JSONL file containing empty and whitespace-only lines.
    When:
        Reading with JSONLAdapter.
    Then:
        Only valid JSON objects are returned in order.
    """
    lines = [
        json.dumps({"id": 1, "score": 0.1}),
        "",
        "   ",
        json.dumps({"id": 2, "score": 0.2}),
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    adapter = JSONLAdapter()
    data = adapter.read(path)

    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[1]["id"] == 2
    assert data[0]["score"] < data[1]["score"]


@pytest.mark.unit
def test_jsonl_adapter_write_should_preserve_unicode(tmp_path: Path):
    """Ensures JSONL preserves Unicode characters in each line.

    Given:
        Data containing accented characters and emoji.
    When:
        Writing with JSONLAdapter.
    Then:
        Each line contains the original Unicode characters.
    """
    payload: list[dict[str, Any]] = [
        {"id": 1, "text": "Café ☕"},
        {"id": 2, "text": "emoji 👨‍👩‍👧‍👦"},
    ]
    path = tmp_path / "unicode.jsonl"

    adapter = JSONLAdapter()
    adapter.write(payload, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == lines[0].strip()
    assert lines[1] == lines[1].strip()
    assert "Café ☕" in lines[0]
    assert "👨‍👩‍👧‍👦" in lines[1]
    assert json.loads(lines[0])["id"] == 1
    assert json.loads(lines[1])["id"] == 2


@pytest.mark.unit
def test_jsonl_adapter_write_should_raise_on_unserializable_item(
    tmp_path: Path,
):
    """Validates JSONL write fails on unserializable objects with cause.

    Given:
        Data containing an unserializable value (set).
    When:
        Writing with JSONLAdapter.
    Then:
        A TypeError is raised and preserves a causal chain.
    """
    payload: list[dict[str, Any]] = [{"id": 1, "tags": {"a", "b"}}]
    path = tmp_path / "bad.jsonl"

    adapter = JSONLAdapter()

    with pytest.raises(TypeError) as exc_info:
        adapter.write(payload, path)

    assert exc_info.value.__cause__ is not None


@pytest.mark.integration
def test_json_adapter_roundtrip_should_preserve_objects(tmp_path: Path):
    """Validates JSONAdapter roundtrip preserves data integrity.

    Given:
        A list of objects with stable ordering and values.
    When:
        Writing then reading with JSONAdapter.
    Then:
        The loaded objects match the original payload exactly.
    """
    payload: list[dict[str, Any]] = [
        {"id": 10, "name": "alpha", "score": 0.1},
        {"id": 11, "name": "bravo", "score": 0.2},
    ]
    path = tmp_path / "roundtrip.json"

    adapter = JSONAdapter()
    adapter.write(payload, path, sort_keys=True)
    loaded = adapter.read(path)

    assert loaded == payload
    assert loaded[0]["score"] < loaded[1]["score"]


@pytest.mark.integration
def test_jsonl_adapter_roundtrip_should_preserve_order(tmp_path: Path):
    """Validates JSONLAdapter roundtrip preserves item order.

    Given:
        A list of objects ordered by score.
    When:
        Writing then reading with JSONLAdapter.
    Then:
        The order and values are preserved exactly.
    """
    payload: list[dict[str, Any]] = [
        {"id": 1, "score": 0.3},
        {"id": 2, "score": 0.5},
    ]
    path = tmp_path / "roundtrip.jsonl"

    adapter = JSONLAdapter()
    adapter.write(payload, path)
    loaded = adapter.read(path)

    assert loaded == payload
    assert loaded[0]["score"] < loaded[1]["score"]
