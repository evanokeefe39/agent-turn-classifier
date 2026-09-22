"""Minimal CLI. Only `tracer` exists today; the real surface lands with v0.1.0."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atc", description="agent-turn-classifier")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tracer", help="run the tracer bullet end to end (no key, no network)")

    args = parser.parse_args(argv)

    if args.cmd == "tracer":
        from . import tracer

        print(json.dumps(tracer.run_tracer(), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
