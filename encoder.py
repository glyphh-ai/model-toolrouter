"""
Custom encoder for the DevOps tool router model.

Exports:
  ENCODER_CONFIG — EncoderConfig with intent-focused routing layers
  encode_query(query) — converts NL text to a Concept dict for similarity search
  entry_to_record(entry) — converts a JSONL entry to an encodable record

Uses Glyphh HDC primitives:
  - Two-layer architecture: intent (verb/object/domain) + context (keywords)
  - Verb carries highest weight for action disambiguation
  - Domain layer separates release/test/build/docker concerns
  - Lexicons on roles for NL query matching
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.7,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="verb",
                            similarity_weight=1.0,
                            lexicons=[
                                "create", "build", "run", "deploy", "release",
                                "check", "monitor", "delete", "clean", "restart",
                                "merge", "tag", "ship", "trigger", "execute",
                            ],
                        ),
                        Role(
                            name="object",
                            similarity_weight=0.8,
                            lexicons=[
                                "branch", "tag", "tests", "merge", "workflow",
                                "build", "sdk", "runtime", "platform", "studio",
                                "docker", "container", "logs", "pipeline",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="scope",
                    roles=[
                        Role(
                            name="domain",
                            similarity_weight=1.0,
                            lexicons=[
                                "release", "test", "build", "docker",
                                "ci", "cd", "deploy", "pipeline",
                            ],
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.3,
            segments=[
                Segment(
                    name="keywords",
                    roles=[
                        Role(
                            name="keywords",
                            similarity_weight=1.0,
                            lexicons=[
                                "release", "branch", "tag", "merge", "test",
                                "build", "deploy", "docker", "restart", "logs",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# NL extraction helpers
# ---------------------------------------------------------------------------

_VERB_MAP = {
    "create": "build", "build": "build", "init": "build", "start": "build",
    "prepare": "build", "setup": "build", "make": "build", "cut": "build",
    "rebuild": "build", "recreate": "build", "nuke": "build",
    "run": "deploy", "execute": "deploy", "trigger": "deploy",
    "deploy": "deploy", "release": "deploy", "ship": "deploy",
    "publish": "deploy", "push": "deploy", "tag": "deploy",
    "merge": "deploy", "integrate": "deploy",
    "check": "query", "status": "query", "monitor": "query",
    "get": "query", "show": "query", "list": "query", "is": "query",
    "logs": "query", "tail": "query", "view": "query",
    "delete": "manage", "remove": "manage", "clean": "manage",
    "cleanup": "manage", "restart": "manage", "bounce": "manage",
    "stop": "manage",
}

_KNOWN_OBJECTS = [
    "sdk", "runtime", "platform", "studio",
    "branch", "tag", "tests", "test", "merge",
    "build", "workflow", "pipeline", "ci", "wheels",
    "version", "suite", "regression", "release",
    "docker", "container", "containers", "logs",
]

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it",
    "do", "can", "please", "now", "up", "my", "our", "me",
    "and", "or", "of", "with", "from", "this", "that",
    "how", "what", "when", "where", "which", "who",
}


def _extract_verb(words: list[str]) -> str:
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _VERB_MAP:
            return _VERB_MAP[clean]
    return "query"


def _extract_object(words: list[str]) -> str:
    for obj in _KNOWN_OBJECTS:
        if obj in words:
            return obj
    for w in reversed(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean and clean not in _STOP_WORDS:
            return clean
    return "general"


def _infer_domain(words: list[str]) -> str:
    text = " ".join(words)
    if any(w in text for w in [
        "docker", "container", "compose", "rebuild", "restart",
        "bounce", "logs", "nuke", "image",
    ]):
        return "docker"
    if any(w in text for w in [
        "test", "pytest", "regression", "suite", "unit", "property",
    ]):
        return "test"
    if any(w in text for w in [
        "build", "wheel", "ci", "workflow", "pipeline", "actions",
        "compile", "package", "status",
    ]):
        return "build"
    if any(w in text for w in [
        "release", "tag", "branch", "merge", "ship", "version",
        "deploy", "publish",
    ]):
        return "release"
    return "release"


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query about devops into a Concept-compatible dict."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower())
    words = cleaned.split()

    verb = _extract_verb(words)
    obj = _extract_object(words)
    domain = _infer_domain(words)
    keywords = " ".join(w for w in words if w not in _STOP_WORDS)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "verb": verb,
            "object": obj,
            "domain": domain,
            "keywords": keywords,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL entry → encodable record + metadata
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL entry to an encodable record with metadata."""
    question = entry.get("question", "").lower()
    tool_id = entry.get("tool_id", "")
    verb = entry.get("verb", "query")
    obj = entry.get("object", "general")
    domain = entry.get("domain", "release")
    kw_list = entry.get("keywords", [])
    kw_str = " ".join(kw_list) if isinstance(kw_list, list) else str(kw_list)

    slug = re.sub(r"[^a-z0-9]+", "_", question).strip("_")[:40]

    return {
        "concept_text": question,
        "attributes": {
            "verb": verb,
            "object": obj,
            "domain": domain,
            "keywords": kw_str,
        },
        "metadata": {
            "tool_id": tool_id,
            "description": entry.get("description", ""),
            "original_question": question,
        },
    }
