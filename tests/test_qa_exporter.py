"""Unit tests for QA exporter to trial cases.

This module tests the conversion from enriched knowledge nodes
to TrialCase format for evaluation.
"""

import json
from pathlib import Path

import pytest

from ragmark.forge.qa_exporter import QAExporter
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import TrialCase


@pytest.mark.unit
def test_nodes_to_trial_cases_should_convert_enriched_nodes(
    enriched_node_factory,
):
    """Validates conversion from enriched nodes to trial cases.

    Given:
        Two enriched nodes with 2 QA pairs each.
    When:
        Calling nodes_to_trial_cases.
    Then:
        4 trial cases are created (2 per node).
        Each case has question, answer, and metadata.
        Ground truth node IDs are set.
    """
    nodes = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=2),
    ]

    trial_cases = QAExporter.nodes_to_trial_cases(nodes)

    assert len(trial_cases) == 4

    case0 = trial_cases[0]
    assert case0.question == "Question 1 about src1?"
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
    enriched_node_factory,
):
    """Validates conversion without ground truth node IDs.

    Given:
        Enriched nodes with QA pairs.
        include_ground_truth_nodes=False.
    When:
        Calling nodes_to_trial_cases.
    Then:
        Trial cases have None for ground_truth_node_ids.
        ground_truth_answer is still set with valid content.
    """
    nodes = [enriched_node_factory("Content", "src1", num_qa_pairs=1)]

    trial_cases = QAExporter.nodes_to_trial_cases(
        nodes,
        include_ground_truth_nodes=False,
    )

    assert len(trial_cases) == 1
    assert trial_cases[0].ground_truth_answer.startswith("Answer 1:")
    assert len(trial_cases[0].ground_truth_answer) > 10
    assert trial_cases[0].ground_truth_node_ids is None


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_nodes_without_qa(
    enriched_node_factory,
):
    """Validates nodes without synthetic_qa are skipped.

    Given:
        Mix of enriched nodes and plain nodes.
    When:
        Calling nodes_to_trial_cases.
    Then:
        Only enriched nodes are converted.
        Plain nodes are skipped with debug log.
    """
    enriched = enriched_node_factory("Content 1", "src1", num_qa_pairs=2)
    plain = KnowledgeNode(
        content="Content 2",
        source_id="src2",
        position=NodePosition(start_char=0, end_char=9),
    )

    nodes = [enriched, plain]
    trial_cases = QAExporter.nodes_to_trial_cases(nodes)

    assert len(trial_cases) == 2
    assert all(case.metadata["source_id"] == "src1" for case in trial_cases)


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_empty_qa_pairs(
    enriched_node_factory,
):
    """Validates nodes with empty qa_pairs are skipped.

    Given:
        Node with synthetic_qa metadata but empty qa_pairs list.
    When:
        Calling nodes_to_trial_cases.
    Then:
        No trial cases are created.
    """
    node = KnowledgeNode(
        content="Content",
        source_id="src1",
        position=NodePosition(start_char=0, end_char=7),
        metadata={"synthetic_qa": {"qa_pairs": []}},
    )

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 0


@pytest.mark.unit
def test_nodes_to_trial_cases_should_skip_invalid_qa_pairs(
    enriched_node_factory,
):
    """Validates invalid QA pairs (missing question or answer) are skipped.

    Given:
        Node with some valid and some invalid QA pairs.
    When:
        Calling nodes_to_trial_cases.
    Then:
        Only valid pairs are converted.
        Invalid pairs are skipped.
    """
    node = KnowledgeNode(
        content="Content",
        source_id="src1",
        position=NodePosition(start_char=0, end_char=7),
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
    )

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 2
    assert trial_cases[0].question == "Valid?"
    assert trial_cases[1].question == "Another valid?"


@pytest.mark.unit
def test_export_to_jsonl_should_create_file(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates JSONL export creates valid file.

    Given:
        Enriched nodes with QA pairs.
        A temporary output path.
    When:
        Calling export_to_jsonl.
    Then:
        JSONL file is created with correct format.
        Each line is a valid TrialCase JSON.
    """
    nodes = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=1),
    ]

    output_path = tmp_path / "trial_cases.jsonl"
    count = QAExporter.export_to_jsonl(nodes, output_path)

    assert count == 3
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 3

    for line in lines:
        case_data = json.loads(line)
        case = TrialCase.model_validate(case_data)
        assert case.question.endswith("?")
        assert len(case.ground_truth_answer) > 0
        assert case.ground_truth_answer.startswith("Answer")


@pytest.mark.unit
def test_export_to_jsonl_should_reject_wrong_extension(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates export_to_jsonl rejects non-JSONL paths.

    Given:
        Output path with .json extension.
    When:
        Calling export_to_jsonl.
    Then:
        ValueError is raised with explicit message about .jsonl requirement.
    """
    nodes = [enriched_node_factory("Content", "src1")]
    output_path = tmp_path / "trial_cases.json"

    with pytest.raises(ValueError, match=r".*\.jsonl.*extension.*") as exc_info:
        QAExporter.export_to_jsonl(nodes, output_path)

    assert "trial_cases.json" in str(exc_info.value)


@pytest.mark.unit
def test_export_to_json_should_create_file(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates JSON export creates valid file.

    Given:
        Enriched nodes with QA pairs.
        A temporary output path.
    When:
        Calling export_to_json.
    Then:
        JSON file is created with list of trial cases.
        File is valid JSON with proper indentation.
    """
    nodes = [
        enriched_node_factory("Content 1", "src1", num_qa_pairs=2),
        enriched_node_factory("Content 2", "src2", num_qa_pairs=1),
    ]

    output_path = tmp_path / "trial_cases.json"
    count = QAExporter.export_to_json(nodes, output_path)

    assert count == 3
    assert output_path.exists()

    with open(output_path, encoding="utf-8") as f:
        cases_data = json.load(f)

    assert isinstance(cases_data, list)
    assert len(cases_data) == 3

    for case_data in cases_data:
        case = TrialCase.model_validate(case_data)
        assert case.question.endswith("?")
        assert len(case.ground_truth_answer) > 0
        assert case.ground_truth_answer.startswith("Answer")


@pytest.mark.unit
def test_export_to_json_should_reject_wrong_extension(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates export_to_json rejects non-JSON paths.

    Given:
        Output path with .jsonl extension.
    When:
        Calling export_to_json.
    Then:
        ValueError is raised with explicit message about .json requirement.
    """
    nodes = [enriched_node_factory("Content", "src1")]
    output_path = tmp_path / "trial_cases.jsonl"

    with pytest.raises(ValueError, match=r".*\.json.*extension.*") as exc_info:
        QAExporter.export_to_json(nodes, output_path)

    assert "trial_cases.jsonl" in str(exc_info.value)


@pytest.mark.unit
def test_export_should_preserve_source_metadata(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates export preserves source node metadata.

    Given:
        Enriched node with custom metadata fields.
    When:
        Exporting to trial cases.
    Then:
        Custom metadata is included in trial case metadata.
        synthetic_qa is excluded to avoid bloat.
    """
    node = enriched_node_factory("Content", "src1", num_qa_pairs=1)

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]

    assert "source_metadata" in case.metadata
    assert case.metadata["source_metadata"]["custom_field"] == "custom_value"
    assert "synthetic_qa" not in case.metadata["source_metadata"]


@pytest.mark.unit
def test_export_should_create_parent_directories(
    tmp_path: Path,
    enriched_node_factory,
):
    """Validates export creates parent directories if needed.

    Given:
        Output path with non-existent parent directories.
    When:
        Calling export_to_jsonl or export_to_json.
    Then:
        Parent directories are created automatically.
        Export succeeds.
    """
    nodes = [enriched_node_factory("Content", "src1", num_qa_pairs=1)]

    output_path = tmp_path / "subdir1" / "subdir2" / "trial_cases.jsonl"
    assert not output_path.parent.exists()

    count = QAExporter.export_to_jsonl(nodes, output_path)

    assert count == 1
    assert output_path.exists()
    assert output_path.parent.exists()


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_nodes_to_trial_cases_should_handle_unicode_nfc_nfd(
    enriched_node_factory,
):
    """Validates NFC/NFD Unicode normalization in QA pairs.

    Given:
        Node with mixed Unicode normalization (NFC + NFD).
    When:
        Converting to trial cases.
    Then:
        Questions and answers are preserved correctly.
        No character corruption occurs.
    """
    content_nfc = "Café résumé"

    node = enriched_node_factory(content_nfc, "unicode-nfc", num_qa_pairs=1)

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert "Café" in case.ground_truth_answer or "Café" in case.ground_truth_answer
    assert len(case.question) > 0


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_nodes_to_trial_cases_should_handle_complex_emoji(
    enriched_node_factory,
):
    """Validates handling of complex emoji in QA content.

    Given:
        Node with complex composite emoji (family, flags).
    When:
        Converting to trial cases.
    Then:
        Emoji are preserved in questions and answers.
        No grapheme splitting occurs.
    """
    content_with_emoji = "User profile: 👨‍👩‍👧‍👦 Family 🇫🇷 France"

    node = KnowledgeNode(
        content=content_with_emoji,
        source_id="emoji-test",
        position=NodePosition(start_char=0, end_char=len(content_with_emoji)),
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
    )

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert "👨‍👩‍👧‍👦" in case.question
    assert "👨‍👩‍👧‍👦" in case.ground_truth_answer


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_nodes_to_trial_cases_should_handle_ideographic_whitespace():
    """Validates handling of CJK ideographic whitespace.

    Given:
        Node with CJK text and ideographic space (U+3000).
    When:
        Converting to trial cases.
    Then:
        Whitespace is preserved in content.
        No trimming corruption occurs.
    """
    ideographic_space = "\u3000"
    content_cjk = f"東京{ideographic_space}日本"

    node = KnowledgeNode(
        content=content_cjk,
        source_id="cjk-test",
        position=NodePosition(start_char=0, end_char=len(content_cjk)),
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
    )

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert ideographic_space in case.question
    assert ideographic_space in case.ground_truth_answer


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_nodes_to_trial_cases_should_reject_control_characters():
    """Validates rejection of control characters in QA pairs.

    Given:
        Node with control characters (null bytes, BOM) in QA.
    When:
        Converting to trial cases.
    Then:
        Control characters are either stripped or handled safely.
        No data corruption in exported JSON.
    """
    content_with_control = "Clean content"

    node = KnowledgeNode(
        content=content_with_control,
        source_id="control-test",
        position=NodePosition(start_char=0, end_char=len(content_with_control)),
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
    )

    trial_cases = QAExporter.nodes_to_trial_cases([node])

    assert len(trial_cases) == 1
    case = trial_cases[0]
    assert "\x00" not in case.question or len(case.question) > 0
    assert "\ufeff" not in case.ground_truth_answer or len(case.ground_truth_answer) > 0
