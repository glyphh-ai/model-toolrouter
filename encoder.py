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
    # build verbs
    "create": "build", "build": "build", "init": "build", "start": "build",
    "prepare": "build", "setup": "build", "make": "build", "cut": "build",
    "rebuild": "build", "recreate": "build", "nuke": "build", "spin": "build",
    # deploy verbs
    "run": "deploy", "execute": "deploy", "trigger": "deploy",
    "deploy": "deploy", "release": "deploy", "ship": "deploy",
    "publish": "deploy", "push": "deploy", "tag": "deploy",
    "merge": "deploy", "integrate": "deploy", "fold": "deploy",
    "stamp": "deploy", "finish": "deploy",
    # query verbs
    "check": "query", "status": "query", "monitor": "query",
    "get": "query", "show": "query", "list": "query", "is": "query",
    "logs": "query", "tail": "query", "view": "query", "doing": "query",
    # manage verbs
    "delete": "manage", "remove": "manage", "clean": "manage",
    "cleanup": "manage", "restart": "manage", "bounce": "manage",
    "stop": "manage", "rid": "manage",
}

# Multi-word phrases that map to (verb, object, domain) tuples.
# Checked before single-word extraction for idiom coverage.
_PHRASE_MAP = {
    "ship it": ("deploy", "release", "release"),
    "kick the tires": ("deploy", "tests", "test"),
    "kick tires": ("deploy", "tests", "test"),
    "fresh coat of paint": ("build", "runtime", "docker"),
    "fresh coat": ("build", "runtime", "docker"),
    "put a bow": ("deploy", "tag", "release"),
    "start fresh": ("build", "docker", "docker"),
    "nuke everything": ("build", "docker", "docker"),
    "is everything green": ("query", "workflow", "build"),
    "everything green": ("query", "workflow", "build"),
    "all green": ("query", "workflow", "build"),
    "nothing is broken": ("deploy", "tests", "test"),
    "nothing broken": ("deploy", "tests", "test"),
    "make sure": ("deploy", "tests", "test"),
    "check on things": ("query", "workflow", "build"),
    "what boxes": ("query", "docker", "docker"),
    "boxes are up": ("query", "docker", "docker"),
    "get rid": ("manage", "branch", "release"),
    "finish the release": ("manage", "branch", "release"),
    "start the release": ("build", "branch", "release"),
    "fold": ("deploy", "merge", "release"),
}

_KNOWN_OBJECTS = [
    # Most specific first — order matters for first-match
    "logs", "tests", "test", "workflow",
    "sdk", "runtime", "platform", "studio",
    "branch", "tag", "merge",
    "build", "ci", "wheels",
    "version", "suite", "regression", "release",
    "docker", "container", "containers",
]

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it",
    "do", "can", "please", "now", "up", "my", "our", "me",
    "and", "or", "of", "with", "from", "this", "that",
    "how", "what", "when", "where", "which", "who",
}

# Qualifiers that indicate out-of-scope intent even when verb/object match.
# These are contexts our tool catalog doesn't support — e.g. "hotfix branch"
# matches create_branch structurally, but hotfix is a different workflow.
_OUT_OF_SCOPE_QUALIFIERS = {
    "hotfix", "staging", "feature", "canary", "rollback",
    "security", "scan", "audit", "nginx", "database", "db",
    "redis", "kafka", "queue", "scale", "scaling", "instances",
    "replicas", "load", "balancer",
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
    # Synonym mapping for objects not in the canonical list
    _obj_synonyms = {
        "backend": "runtime", "server": "runtime", "service": "runtime",
        "frontend": "studio", "app": "studio", "web": "studio",
        "boxes": "docker", "images": "docker",
        "tires": "tests", "broken": "tests",
        "changes": "merge", "pr": "merge",
        "rebuild": "docker", "deploy": "release",
        "pipeline": "workflow",
    }
    for w in words:
        if w in _obj_synonyms:
            return _obj_synonyms[w]
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
    text = " ".join(words)

    # Check multi-word phrases first (idiom coverage)
    verb, obj, domain = None, None, None
    for phrase, (pv, po, pd) in _PHRASE_MAP.items():
        if phrase in text:
            verb, obj, domain = pv, po, pd
            break

    if verb is None:
        verb = _extract_verb(words)
    if obj is None:
        obj = _extract_object(words)
    if domain is None:
        domain = _infer_domain(words)

    keywords = " ".join(w for w in words if w not in _STOP_WORDS)

    # Check for out-of-scope qualifiers — words that indicate the query
    # targets a context our tool catalog doesn't cover.
    oos_hits = _OUT_OF_SCOPE_QUALIFIERS & set(words)

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "verb": verb,
            "object": obj,
            "domain": domain,
            "keywords": keywords,
        },
        "out_of_scope_qualifiers": oos_hits,
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
