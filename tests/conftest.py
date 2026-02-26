"""Shared fixtures for tool router tests."""

import json
from pathlib import Path

import pytest

from encoder import ENCODER_CONFIG, encode_query, entry_to_record, ToolNameSidecar
from glyphh.core.types import Concept
from glyphh.encoder import Encoder


@pytest.fixture(scope="session")
def encoder():
    """Session-scoped encoder instance."""
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="session")
def sidecar():
    """Session-scoped sidecar instance."""
    data_path = str(Path(__file__).parent.parent / "data" / "exemplars.jsonl")
    return ToolNameSidecar(data_path)


@pytest.fixture(scope="session")
def exemplar_entries():
    """Load all exemplar entries from JSONL."""
    data_path = Path(__file__).parent.parent / "data" / "exemplars.jsonl"
    entries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


@pytest.fixture(scope="session")
def exemplar_glyphs(encoder, exemplar_entries):
    """Encode all exemplars into glyphs."""
    glyphs = []
    for entry in exemplar_entries:
        record = entry_to_record(entry)
        concept = Concept(
            name=f"tool_{entry['tool_id']}",
            attributes=record["attributes"],
        )
        glyphs.append((encoder.encode(concept), entry))
    return glyphs


@pytest.fixture(scope="session")
def benchmark_queries():
    """Load benchmark queries from JSON."""
    path = Path(__file__).parent.parent / "benchmark" / "queries.json"
    with open(path) as f:
        data = json.load(f)
    return data["queries"]
