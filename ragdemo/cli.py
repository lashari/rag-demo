"""Command-line interface:  python -m ragdemo.cli ask "your question"."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RagPipeline

DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ragdemo", description="Minimal hybrid RAG demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Index a file/directory and print stats")
    p_ing.add_argument("path", nargs="?", default=DEFAULT_DATA)

    p_ask = sub.add_parser("ask", help="Index ./data then answer a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--data", default=DEFAULT_DATA)
    p_ask.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args(argv)
    rag = RagPipeline()

    if args.cmd == "ingest":
        print(json.dumps(rag.ingest_path(args.path), indent=2))
        return

    rag.ingest_path(args.data)
    out = rag.answer(args.question, top_k=args.top_k)
    print("\nAnswer:\n" + out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['source']} (score {s['score']}) — {s['preview']}")
    print(f"\n(embedder + generator: {rag.embedder.name} / {out['generator']})")


if __name__ == "__main__":
    main()
