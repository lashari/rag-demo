"""Offline tests for the command-line interface (ragdemo/cli.py)."""
import json
import runpy
import sys

import pytest

from ragdemo import cli


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "rrf.md").write_text(
        "Reciprocal rank fusion merges several ranked lists into one. "
        "Each document scores the sum of one over k plus its rank in every list. ",
        encoding="utf-8",
    )
    (tmp_path / "quant.txt").write_text(
        "Quantization stores vectors as int8 to cut memory and speed up search. ",
        encoding="utf-8",
    )
    (tmp_path / "ignored.csv").write_text("not,indexed\n", encoding="utf-8")
    return tmp_path


def source_lines(output):
    return [line for line in output.splitlines() if line.startswith("  [")]


def test_ingest_default_data_dir_prints_json_stats(capsys):
    cli.main(["ingest"])
    stats = json.loads(capsys.readouterr().out)
    assert stats["chunks_indexed"] > 0
    assert stats["embedder"]


def test_ingest_custom_path_counts_only_md_and_txt(docs, capsys):
    cli.main(["ingest", str(docs)])
    stats = json.loads(capsys.readouterr().out)
    assert stats["chunks_indexed"] == 2  # one chunk per file; .csv is skipped


def test_ingest_single_file(docs, capsys):
    cli.main(["ingest", str(docs / "rrf.md")])
    assert json.loads(capsys.readouterr().out)["chunks_indexed"] == 1


def test_ask_prints_answer_sources_and_backend(capsys):
    cli.main(["ask", "What is reciprocal rank fusion?"])
    out = capsys.readouterr().out
    assert "\nAnswer:\n" in out
    assert "\nSources:" in out
    assert source_lines(out)
    assert out.rstrip().endswith("/ extractive)")


def test_ask_uses_data_option(docs, capsys):
    cli.main(["ask", "How does reciprocal rank fusion merge ranked lists?",
              "--data", str(docs)])
    lines = source_lines(capsys.readouterr().out)
    assert lines[0].startswith("  [1] rrf.md (score ")
    assert all("ignored.csv" not in line for line in lines)


def test_ask_top_k_limits_sources(docs, capsys):
    cli.main(["ask", "quantization", "--data", str(docs), "--top-k", "1"])
    assert len(source_lines(capsys.readouterr().out)) == 1


@pytest.mark.parametrize("argv", [[], ["ask"], ["ask", "q", "--top-k", "x"], ["nope"]])
def test_bad_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "usage: ragdemo" in capsys.readouterr().err


def test_module_entry_point(monkeypatch, docs, capsys):
    # Covers `python -m ragdemo.cli ...` (the __main__ guard).
    monkeypatch.setattr(sys, "argv", ["ragdemo", "ingest", str(docs)])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("ragdemo.cli", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["chunks_indexed"] == 2
