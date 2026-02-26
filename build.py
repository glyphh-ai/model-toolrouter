#!/usr/bin/env python3
"""
Build the SaaS tool router model from JSONL exemplar data.

Usage:
    python build.py
    python build.py --output path/to/output.glyphh
"""

import argparse
import json
import sys
from pathlib import Path

from encoder import entry_to_record

MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR / "data"
DEFAULT_OUTPUT = MODEL_DIR / "toolrouter.glyphh"

JSONL_FILES = [
    "exemplars.jsonl",
]


def load_all_jsonl(data_dir: Path) -> list[dict]:
    entries = []
    for filename in JSONL_FILES:
        path = data_dir / filename
        if not path.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue
        count = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError:
                    print(f"  Warning: bad JSON at {filename}:{lineno}, skipping")
        print(f"  {filename}: {count} entries")
    return entries


def build(output_path: Path | None = None) -> None:
    output = output_path or DEFAULT_OUTPUT

    print("Loading JSONL exemplar data...")
    entries = load_all_jsonl(DATA_DIR)
    if not entries:
        print("Error: No entries found.")
        sys.exit(1)

    print(f"\nConverting {len(entries)} entries to records...")
    records = [entry_to_record(e) for e in entries]

    # Show tool distribution
    tool_counts: dict[str, int] = {}
    for r in records:
        tid = r["metadata"]["tool_id"]
        tool_counts[tid] = tool_counts.get(tid, 0) + 1
    print(f"\nTool distribution ({len(tool_counts)} tools):")
    for tid, count in sorted(tool_counts.items()):
        print(f"  {tid}: {count} exemplars")

    print(f"\nTotal records: {len(records)}")
    print(f"\nReady to package as {output}")
    print("(Packaging requires the Glyphh runtime SDK)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SaaS tool router model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
