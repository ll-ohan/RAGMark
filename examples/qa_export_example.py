"""Example: Export synthetic QA pairs to evaluation trial cases.

This script demonstrates how to:
1. Load enriched knowledge nodes with synthetic QA metadata
2. Convert them to TrialCase format using adapters
3. Export to JSONL or JSON for evaluation

Usage:
    python examples/qa_export_example.py
"""

from pathlib import Path

from ragmark.adapters.formats.json_adapter import JSONAdapter
from ragmark.adapters.formats.jsonl_adapter import JSONLAdapter
from ragmark.adapters.transformers.qa_adapter import NodeToTrialCaseAdapter
from ragmark.schemas.documents import KnowledgeNode, NodePosition


def create_sample_enriched_nodes() -> list[KnowledgeNode]:
    """Create sample enriched nodes with synthetic QA metadata.

    Returns:
        List of enriched knowledge nodes.
    """
    # Simulate enriched nodes with synthetic_qa metadata
    # (normally these would come from LLMQuestionGenerator)

    node1 = KnowledgeNode(
        content="Python is a high-level, interpreted programming language. "
        "It was created by Guido van Rossum and first released in 1991. "
        "Python emphasizes code readability with significant whitespace.",
        source_id="python_intro.txt",
        position=NodePosition(
            start_char=0, end_char=200, page=1, section="Introduction"
        ),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {
                        "question": "What type of programming language is Python?",
                        "answer": "Python is a high-level, interpreted programming language.",
                        "confidence": 0.95,
                    },
                    {
                        "question": "Who created Python and when was it first released?",
                        "answer": "Python was created by Guido van Rossum and first released in 1991.",
                        "confidence": 0.92,
                    },
                    {
                        "question": "What does Python emphasize in its design?",
                        "answer": "Python emphasizes code readability with significant whitespace.",
                        "confidence": 0.88,
                    },
                ],
                "generated_at": "2024-01-15T10:30:00Z",
                "model": "llama-3-8b-instruct.gguf",
                "num_questions_requested": 3,
                "num_questions_validated": 3,
                "batch_id": "batch_001",
            },
            "chunk_type": "introduction",
        },
        dense_vector=None,
        sparse_vector=None,
    )

    node2 = KnowledgeNode(
        content="FastAPI is a modern, fast web framework for building APIs with Python. "
        "It is based on standard Python type hints and provides automatic API documentation. "
        "FastAPI achieves high performance comparable to NodeJS and Go.",
        source_id="fastapi_overview.txt",
        position=NodePosition(
            start_char=0, end_char=180, page=1, section="Introduction"
        ),
        metadata={
            "synthetic_qa": {
                "qa_pairs": [
                    {
                        "question": "What is FastAPI?",
                        "answer": "FastAPI is a modern, fast web framework for building APIs with Python.",
                        "confidence": 0.93,
                    },
                    {
                        "question": "What does FastAPI provide for API development?",
                        "answer": "FastAPI provides automatic API documentation based on standard Python type hints.",
                        "confidence": 0.90,
                    },
                ],
                "generated_at": "2024-01-15T10:30:05Z",
                "model": "llama-3-8b-instruct.gguf",
                "num_questions_requested": 2,
                "num_questions_validated": 2,
                "batch_id": "batch_001",
            },
            "chunk_type": "overview",
        },
        dense_vector=None,
        sparse_vector=None,
    )

    return [node1, node2]


def main():
    """Run the QA export example."""
    print("🔄 Creating sample enriched nodes...")
    nodes = create_sample_enriched_nodes()
    print(f"✅ Created {len(nodes)} enriched nodes")

    # Convert to trial cases using NodeToTrialCaseAdapter
    print("\n🔄 Converting nodes to trial cases...")
    qa_adapter = NodeToTrialCaseAdapter(include_ground_truth_nodes=True)
    trial_cases = qa_adapter.adapt_many(nodes)
    print(f"✅ Created {len(trial_cases)} trial cases")

    # Display first trial case as example
    print("\n📋 Sample trial case:")
    print(f"  Question: {trial_cases[0].question}")
    print(f"  Answer: {trial_cases[0].ground_truth_answer}")
    print(f"  Source Node ID: {trial_cases[0].ground_truth_node_ids}")
    print(f"  Metadata: {trial_cases[0].metadata}")

    # Export to JSONL
    output_dir = Path("output/trial_cases")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "qa_trial_cases.jsonl"
    print(f"\n💾 Exporting to JSONL: {jsonl_path}")
    jsonl_adapter = JSONLAdapter()
    cases_data = [case.model_dump() for case in trial_cases]
    jsonl_adapter.write(cases_data, jsonl_path)
    print(f"✅ Exported {len(trial_cases)} trial cases to JSONL")

    # Export to JSON (pretty-printed)
    json_path = output_dir / "qa_trial_cases.json"
    print(f"\n💾 Exporting to JSON: {json_path}")
    json_adapter = JSONAdapter()
    json_adapter.write(cases_data, json_path, indent=2)
    print(f"✅ Exported {len(trial_cases)} trial cases to JSON")

    print("\n✨ Export complete! You can now use these trial cases for evaluation.")
    print("\n📁 Output files:")
    print(f"  - JSONL: {jsonl_path}")
    print(f"  - JSON:  {json_path}")

    # Show how to load them back
    print("\n🔄 Loading trial cases from JSONL...")
    from ragmark.schemas.evaluation import TrialCase

    loaded_cases = TrialCase.load_cases(jsonl_path)
    print(f"✅ Loaded {len(loaded_cases)} trial cases")

    print("\n🎯 Next steps:")
    print("  1. Use these trial cases with the evaluation module")
    print("  2. Run RAG pipeline against these questions")
    print("  3. Compute metrics (recall, MRR, faithfulness, etc.)")


if __name__ == "__main__":
    main()
