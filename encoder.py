"""
Encoder for the SaaS tool router model.

Exports:
  ENCODER_CONFIG — EncoderConfig with intent + semantics layers
  SIDECAR_CONFIG — EncoderConfig for tool name validation (seed=73)
  encode_query(query) — converts NL text to a Concept dict for similarity search
  entry_to_record(entry) — converts a JSONL exemplar entry to a build record
  extract_tool_reference(query) — detects tool name references in queries
  ToolNameSidecar — validates tool references against the real catalog

Architecture:
  NL Extraction — glyphh.intent.IntentExtractor (SDK):
    - Rule-based fast path: phrase patterns → synonym maps → impact ranking
    - HDC fallback for unknown verbs/nouns
    - Domain inference from weighted keyword signals

  Main model (seed=42):
  - Intent layer (0.4): action (lexicon) + target (lexicon) + domain (lexicon)
    → Provides strong signal when extraction is correct (binary match)
  - Semantics layer (0.6): description (BoW) + keywords (BoW)
    → Provides fuzzy HDC matching when extraction is imperfect

  Sidecar model (seed=73):
  - Identity layer (0.7): name_tokens (BoW) — tool name split on underscores
  - Capability layer (0.3): action (lexicon) — primary verb
    → Validates whether a referenced tool name matches a real catalog entry
"""

import hashlib
import re

from glyphh.core.config import (
    EncoderConfig,
    Layer,
    Role,
    Segment,
)
from glyphh.intent import get_extractor

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=[
                                "send", "search", "create", "get", "list",
                                "update", "delete", "reply", "share", "upload",
                                "track", "identify", "assign", "comment",
                                "charge", "refund", "cancel", "subscribe",
                                "draft", "find", "set", "schedule", "book",
                                "notify", "forward", "export", "archive",
                                "none",
                            ],
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.7,
                            lexicons=[
                                "message", "email", "contact", "customer",
                                "ticket", "event", "file", "folder",
                                "channel", "invoice", "subscription", "payment",
                                "deal", "status", "comment", "draft",
                                "metric", "funnel", "user", "label",
                                "dm", "free_time",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="scope",
                    roles=[
                        Role(
                            name="domain",
                            similarity_weight=0.8,
                            lexicons=[
                                "messaging", "email", "crm", "payments",
                                "calendar", "files", "tickets", "analytics",
                                "none",
                            ],
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="text",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="keywords",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Out-of-scope detection (toolrouter-specific suppression)
# ---------------------------------------------------------------------------

# Domains the toolrouter knows about — anything else is normalised to "none"
# so that packs like devops/github/docs don't pollute the intent layer.
_KNOWN_DOMAINS = frozenset({
    "messaging", "email", "crm", "payments",
    "calendar", "files", "tickets", "analytics",
})

# Fallback domain inference: keyword signals tried when IntentExtractor returns
# a domain outside _KNOWN_DOMAINS.  Order matters — first match wins.
_DOMAIN_FALLBACK: list[tuple[str, list[str]]] = [
    ("messaging", ["slack", " #", "/#", "channel", "direct message", " dm "]),
    ("payments",  ["stripe", " invoice", "subscription", " charge", " refund", "billing"]),
    ("tickets",   ["jira", " ticket", " issue", " bug", " story", "sprint"]),
    ("email",     ["email", "inbox", "gmail", "smtp", " cc ", " bcc "]),
    ("crm",       ["crm", "salesforce", "hubspot", "contact record", "deal pipeline"]),
    ("calendar",  ["calendar", "gcal", " event ", " meeting", "standup"]),
    ("files",     ["google drive", "gdrive", " drive", "dropbox", " folder"]),
    ("analytics", ["analytics", "segment", "amplitude", " metric", " funnel", "pageview"]),
]

# Canonical actions from the intent lexicon — used for leading-verb validation.
_KNOWN_ACTIONS = frozenset({
    "send", "search", "create", "get", "list",
    "update", "delete", "reply", "share", "upload",
    "track", "identify", "assign", "comment",
    "charge", "refund", "cancel", "subscribe",
    "draft", "find", "set", "schedule", "book",
    "notify", "forward", "export", "archive",
    "none",
})

# When impact-ranking promotes a late-sentence verb over the opening verb,
# this map lets us recover the correct action from the first word.
_LEADING_VERB_MAP: dict[str, str] = {
    # send / notify
    "send": "send", "email": "send", "mail": "send",
    "notify": "notify", "announce": "send", "post": "send",
    # reply / respond
    "reply": "reply", "respond": "reply",
    # create
    "create": "create", "make": "create", "open": "create", "file": "create",
    "add": "create",
    # get / retrieve
    "get": "get", "fetch": "get", "retrieve": "get",
    "give": "get", "show": "get", "display": "get",
    "pull": "get", "check": "get", "describe": "get",
    "summarize": "get", "summarise": "get", "tell": "get",
    # list
    "list": "list",
    # search / find
    "search": "search", "find": "find", "look": "get",
    # update
    "update": "update", "edit": "update", "change": "update",
    "modify": "update",
    # delete
    "delete": "delete", "remove": "delete",
    # upload / share / draft
    "upload": "upload", "share": "share", "draft": "draft",
    # payments
    "charge": "charge", "refund": "refund",
    "cancel": "cancel", "subscribe": "subscribe",
    # track / identify
    "track": "track", "record": "track", "log": "track",
    "identify": "identify", "mark": "identify",
    # calendar
    "schedule": "schedule", "book": "book",
    # misc
    "assign": "assign", "set": "set",
    "forward": "forward", "export": "export", "archive": "archive",
}

_QUESTION_PATTERNS = [
    "what does", "what is", "what are", "what's the", "whats the",
    "how does", "how do", "how many", "how much",
    "why does", "why is",
    "can you explain", "tell me about",
]

_GREETING_PATTERNS = [
    "hello", "hi there", "hey there", "how are you",
]

_NON_SAAS_PATTERNS = [
    "deploy the", "deploy to", "deploy application",
    "merge the", "merge branch",
    "restart the", "restart server",
    "run the ci", "run ci", "ci pipeline", "ci/cd",
    "pull request", "database server", "dns records",
    # Report / PDF generation — no such tool in this catalog
    "generate a pdf", "generate pdf", "generate report", "pdf report",
    # SMS / phone — not a SaaS tool in this catalog
    "text message to", "send a text message", "send text to",
]

# Payment-specific leading verbs that also pin the domain to 'payments',
# overriding any pack-injected domain the IntentExtractor may have returned.
_LEADING_VERB_DOMAIN: dict[str, str] = {
    "charge": "payments",
    "refund": "payments",
    "cancel": "payments",
    "subscribe": "payments",
}

# Detect "Use the snake_case_name API/tool" adversarial patterns.
# These indicate the user is explicitly naming a (possibly fake) tool via
# its API descriptor — we suppress lexicon signals so the threshold rejects them.
_API_TOOL_RE = re.compile(r'\b\w+(?:_\w+)+\s+(?:api|tool)\b', re.IGNORECASE)


def _split_camel_case(text: str) -> str:
    """Split camelCase identifiers so sendSlackNotification → send Slack Notification.

    This prevents the BoW encoder from seeing the whole identifier as one opaque
    token, allowing shared sub-words (e.g. 'slack') to produce similarity signal.
    """
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


def _is_question(text: str) -> bool:
    """Detect informational questions that should not route to a tool."""
    text_lower = text.lower()
    return (
        any(p in text_lower for p in _QUESTION_PATTERNS)
        or any(text_lower.startswith(p) for p in _GREETING_PATTERNS)
    )


def _is_non_saas(text: str) -> bool:
    """Detect system/devops operations outside the SaaS tool catalog."""
    text_lower = text.lower()
    return any(p in text_lower for p in _NON_SAAS_PATTERNS)


def _has_fake_api_ref(text: str) -> bool:
    """Detect 'snake_case_name API' / 'snake_case_name tool' patterns.

    Queries like 'Call the calendar_reschedule_event API to move my 2pm' or
    'Use the email_forward tool to forward...' reference explicitly named but
    non-existent APIs.  Suppressing lexicon signals prevents the threshold
    from accepting them via a partial domain/action match.
    """
    return bool(_API_TOOL_RE.search(text))


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query into a Concept-compatible dict for tool routing.

    Uses glyphh.intent.IntentExtractor for action/target/domain/keyword extraction.
    Suppresses lexicon signals for informational questions and non-SaaS queries.

    Returns a dict with 'name' and 'attributes' keys. Attributes include:
      action, target, domain — categorical signals from intent extraction
      description, keywords  — BoW text for fuzzy HDC matching
    """
    # Pre-process: split camelCase so sendSlackNotification → send Slack Notification,
    # giving BoW encoding shared sub-words (e.g. 'slack') as similarity signal.
    query_clean = _split_camel_case(query)

    extractor = get_extractor()
    extracted = extractor.extract(query_clean)

    action = extracted["action"]
    target = extracted["target"]
    domain = extracted["domain"]
    bow_text = extracted["keywords"]

    # Normalize domain: pack-injected domains (devops, github, docs, …) are not in
    # the toolrouter lexicon.  Before falling back to "none", try a lightweight
    # keyword scan of the raw query to recover the correct SaaS domain.
    if domain not in _KNOWN_DOMAINS:
        query_lower = query.lower()
        domain = "none"
        for d, signals in _DOMAIN_FALLBACK:
            if any(sig in query_lower for sig in signals):
                domain = d
                break

    # Leading-verb correction: IntentExtractor's impact-ranking can promote a
    # high-rank verb that appears later in the sentence (e.g. "deploy"=11) over
    # the sentence-opening verb (e.g. "send"=5).  Recover the correct action
    # from the first word when the leading verb maps to something different.
    # Also, payment-specific leading verbs pin the domain to 'payments' to
    # override pack-injected domains (e.g. "Refund … update the CRM" → 'crm').
    words = query_clean.lower().split()
    if words:
        leading_action = _LEADING_VERB_MAP.get(words[0])
        if leading_action is not None and leading_action != action:
            action = leading_action
        forced_domain = _LEADING_VERB_DOMAIN.get(words[0])
        if forced_domain is not None:
            domain = forced_domain

    # Suppress lexicon signals for informational questions/greetings,
    # clearly non-SaaS operations (deploy, CI, merge, etc.), and queries
    # that reference a named but non-existent API or tool endpoint.
    if _is_question(query) or _is_non_saas(query) or _has_fake_api_ref(query):
        action = "none"
        domain = "none"

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "action": action,
            "target": target,
            "domain": domain,
            "description": bow_text,
            "keywords": bow_text,
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — JSONL exemplar → build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a JSONL exemplar entry into a record for building/encoding.

    Expects entry keys: tool_id, action, target, domain, keywords, description.
    Returns: {"concept_text": str, "attributes": dict, "metadata": dict}
    """
    keywords = entry.get("keywords", [])
    if isinstance(keywords, list):
        keywords = " ".join(keywords)

    description = entry.get("description", keywords)

    return {
        "concept_text": entry["tool_id"],
        "attributes": {
            "action": entry["action"],
            "target": entry["target"],
            "domain": entry["domain"],
            "description": description,
            "keywords": keywords,
        },
        "metadata": {
            "tool_id": entry["tool_id"],
        },
    }


# ---------------------------------------------------------------------------
# SIDECAR_CONFIG — Tool name validation in independent vector space
# ---------------------------------------------------------------------------

SIDECAR_CONFIG = EncoderConfig(
    dimension=10000,
    seed=73,  # Independent vector space from main model (seed=42)
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="identity",
            similarity_weight=0.7,
            segments=[
                Segment(
                    name="name",
                    roles=[
                        Role(
                            name="name_tokens",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="capability",
            similarity_weight=0.3,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=[
                                "send", "search", "create", "get", "list",
                                "update", "delete", "reply", "share", "upload",
                                "track", "identify", "assign", "comment",
                                "charge", "refund", "cancel", "subscribe",
                                "draft", "find", "set", "schedule", "book",
                                "add", "none",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Tool reference extraction
# ---------------------------------------------------------------------------

# ID prefixes that look like snake_case but are data values, not tool names
_ID_PREFIXES = {"ch_", "sub_", "cus_", "u_", "inv_", "price_", "pi_", "pm_"}

# snake_case pattern: 2+ segments of 2+ lowercase letters each
_SNAKE_CASE_RE = re.compile(r'\b([a-z]{2,}(?:_[a-z]{2,})+)\b')

# camelCase pattern: starts lowercase, has at least one uppercase letter
_CAMEL_CASE_RE = re.compile(r'\b([a-z]+[A-Z][a-zA-Z]*)\b')


def extract_tool_reference(query: str) -> str | None:
    """Extract a tool name reference from a query, if present.

    Detects snake_case (e.g., stripe_pause_subscription) and camelCase
    (e.g., sendSlackNotification) patterns that look like tool names.
    Filters out IDs (ch_xxx, sub_xxx) and event names.

    Returns the extracted reference as space-separated lowercase tokens,
    or None if no tool reference is found.
    """
    for m in _SNAKE_CASE_RE.finditer(query):
        candidate = m.group(1)
        if any(candidate.startswith(p) for p in _ID_PREFIXES):
            continue
        start, end = m.start(), m.end()
        prefix = query[:start].rstrip().lower()
        suffix = query[end:].lstrip().lower()
        if prefix.endswith(("event", " a", " an", " from")):
            continue
        if suffix.startswith(("event", "events")):
            continue
        # Filter funnel-style "X to Y to Z" sequences (analytics queries, not tool names)
        if " to " in suffix[:20]:
            continue
        return candidate.replace("_", " ").lower()

    for m in _CAMEL_CASE_RE.finditer(query):
        candidate = m.group(1)
        suffix = query[m.end():].lstrip().lower()
        # Skip camelCase names that appear as "X tool" or "X api" — these are
        # adversarial patterns where the user is naming a (fake) tool explicitly.
        if suffix.startswith(("tool", "api")):
            continue
        tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', candidate).lower()
        return tokens

    return None


# ---------------------------------------------------------------------------
# ToolNameSidecar — validates tool references against the real catalog
# ---------------------------------------------------------------------------

class ToolNameSidecar:
    """Sidecar model that validates tool name references in an independent
    vector space (seed=73).

    Encodes the real tool names as BoW + action lexicon glyphs.
    When the main model's confidence is uncertain and the query references
    a tool by name, the sidecar checks if that name matches a real tool.
    """

    def __init__(self, exemplar_path: str | None = None):
        from pathlib import Path
        from glyphh.encoder import Encoder
        from glyphh.core.types import Concept

        self.encoder = Encoder(SIDECAR_CONFIG)
        self.tool_glyphs: list[tuple] = []  # (glyph, tool_id)
        self._Concept = Concept

        if exemplar_path is None:
            exemplar_path = str(Path(__file__).parent / "data" / "exemplars.jsonl")
        self._build_catalog(exemplar_path)

    def _build_catalog(self, exemplar_path: str):
        """Build one sidecar glyph per unique tool_id."""
        import json

        seen: set[str] = set()
        with open(exemplar_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tool_id = entry["tool_id"]
                if tool_id in seen:
                    continue
                seen.add(tool_id)

                glyph = self.encoder.encode(self._Concept(
                    name=f"sidecar_{tool_id}",
                    attributes={
                        "name_tokens": tool_id.replace("_", " "),
                        "action": entry.get("action", "none"),
                    },
                ))
                self.tool_glyphs.append((glyph, tool_id))

    def validate(self, tool_ref_tokens: str, threshold: float = 0.55) -> str | None:
        """Check if a tool reference matches a real tool in the catalog.

        Args:
            tool_ref_tokens: Space-separated lowercase tokens from the
                tool reference (output of extract_tool_reference).
            threshold: Minimum sidecar similarity to confirm a match.

        Returns:
            The matching real tool_id if above threshold, else None.
        """
        from glyphh.core.ops import cosine_similarity as cos_sim

        action = get_extractor().extract_action(tool_ref_tokens)

        ref_glyph = self.encoder.encode(self._Concept(
            name="ref",
            attributes={"name_tokens": tool_ref_tokens, "action": action},
        ))

        sidecar_weights = {"name_tokens": 0.7, "action": 0.3}

        ref_roles: dict = {}
        for layer in ref_glyph.layers.values():
            for seg in layer.segments.values():
                ref_roles.update(seg.roles)

        best_score = 0.0
        best_tool: str | None = None

        for glyph, tool_id in self.tool_glyphs:
            cat_roles: dict = {}
            for layer in glyph.layers.values():
                for seg in layer.segments.values():
                    cat_roles.update(seg.roles)

            weighted_sum = 0.0
            weight_total = 0.0
            for rname, w in sidecar_weights.items():
                if rname in ref_roles and rname in cat_roles:
                    sim = float(cos_sim(ref_roles[rname].data, cat_roles[rname].data))
                    weighted_sum += sim * w
                    weight_total += w

            score = weighted_sum / weight_total if weight_total > 0 else 0.0
            if score > best_score:
                best_score = score
                best_tool = tool_id

        return best_tool if best_score >= threshold else None
