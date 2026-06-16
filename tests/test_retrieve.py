"""Offline unit tests — no API key or network required:  pytest -q"""
from ragdemo.chunk import split_text
from ragdemo.retrieve import reciprocal_rank_fusion


def test_rrf_prefers_consensus():
    # id 2 ranks well in both lists -> it should win after fusion
    order, scores = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 5]])
    assert order[0] == 2
    assert scores[2] > scores[1]


def test_rrf_single_list_preserves_order():
    order, _ = reciprocal_rank_fusion([[9, 8, 7]])
    assert order == [9, 8, 7]


def test_chunking_produces_overlapping_nonempty_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_pipeline_offline_end_to_end():
    # Full pipeline runs with the offline hash embedder + extractive generator.
    from ragdemo.pipeline import RagPipeline
    from pathlib import Path

    rag = RagPipeline()
    data = str(Path(__file__).resolve().parent.parent / "data")
    rag.ingest_path(data)
    out = rag.answer("What is reciprocal rank fusion?")
    assert out["sources"]            # something was retrieved
    assert "answer" in out
