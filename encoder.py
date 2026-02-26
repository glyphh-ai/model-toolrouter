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
                                "slack", "email", "gmail", "crm", "stripe",
                                "calendar", "drive", "gdrive", "jira",
                                "analytics", "payment", "messaging",
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
# NL extraction helpers
# ---------------------------------------------------------------------------

# Critical disambiguation phrases only — cases where BoW alone can't resolve
# the ambiguity because competing tools share too many words.
_PHRASE_ACTIONS = [
    # comment vs create: "add a comment" must route to comment, not create
    ("add a comment", "comment"),
    ("add comment", "comment"),
    ("leave a comment", "comment"),
    ("post a comment", "comment"),
    # status vs create: "set status" is its own action
    ("set status", "set"),
    ("set my status", "set"),
    ("set my slack status", "set"),
    # create vs assign: "create a task" must win over "assign it to" in multi-action queries
    ("create a task", "create"),
    ("create a ticket", "create"),
    ("create a bug", "create"),
    ("create a story", "create"),
    ("create a epic", "create"),
    ("create a jira", "create"),
    # assign vs update: "assign it to" must route to assign, not update
    ("assign it to", "assign"),
    ("assign to", "assign"),
    ("assign ticket", "assign"),
    # get vs search: "find the customer/contact/record" is lookup, not search
    ("find the customer", "get"),
    ("find the contact", "get"),
    ("find the record", "get"),
    ("find the contact info", "get"),
    ("look up", "get"),
    ("i need the", "get"),
    ("i need", "get"),
    # identify vs track: trait changes route to identify
    ("upgraded to", "identify"),
    ("downgraded to", "identify"),
    ("switched from", "identify"),
    ("changed plan", "identify"),
    ("changed their", "identify"),
    # payment history is list, not get
    ("payment history", "list"),
    ("billing history", "list"),
    ("invoice history", "list"),
    ("check the invoices", "list"),
    ("check invoices", "list"),
    # search-specific
    ("search for the invoice email", "search"),
    ("search for the email", "search"),
    ("look for", "search"),
    # comment synonyms (phrase patterns)
    ("post a remark", "comment"),
    ("leave a response", "comment"),
    ("add a remark", "comment"),
    ("add a note on", "comment"),
    # --- Synonym verb phrases (perturbation coverage) ---
    # send synonyms
    ("fire off", "send"),
    ("shoot over", "send"),
    ("forward along", "send"),
    # search synonyms
    ("dig through", "search"),
    ("hunt for", "search"),
    ("look through", "search"),
    # create synonyms (after "set status" phrases above)
    ("set up a", "create"),
    ("set up", "create"),
    ("spin up", "create"),
    ("put together a", "create"),
    # reply synonyms
    ("write back", "reply"),
    # draft synonyms
    ("write up", "draft"),
    # charge synonyms
    ("collect from", "charge"),
    # get synonyms
    ("track down", "get"),
    # refund synonyms
    ("return the money", "refund"),
    ("reverse the charge", "refund"),
    ("give back the payment", "refund"),
    ("give back", "refund"),
    # upload synonyms (with article to distinguish from "drop in #channel")
    ("drop in the", "upload"),
    ("drop in a", "upload"),
    ("put in the", "upload"),
    ("put in a", "upload"),
    # share synonyms
    ("let someone see", "share"),
    ("give access", "share"),
    ("open up to", "share"),
    ("read-only access", "share"),
    ("read only access", "share"),
    # passive voice / "should be" patterns
    ("should be sent", "send"),
    ("needs to be sent", "send"),
    # set synonyms
    ("change to", "set"),
]

_PHRASE_TARGETS = [
    ("free time", "free_time"),
    ("available time", "free_time"),
    ("time slot", "free_time"),
    ("time that works", "free_time"),
    ("direct message", "dm"),
    ("invoice email", "email"),
    ("customer details", "customer"),
    ("customer record", "customer"),
    ("customer info", "customer"),
    ("customer profile", "customer"),
    ("client details", "customer"),
    ("contact info", "contact"),
    ("contact record", "contact"),
    ("contact details", "contact"),
    ("individual info", "contact"),
    ("person info", "contact"),
    ("add a comment", "comment"),
    ("add comment", "comment"),
    ("leave a comment", "comment"),
    ("post a remark", "comment"),
    ("leave a response", "comment"),
    ("set status", "status"),
    ("slack status", "status"),
    ("my status", "status"),
    ("payment history", "invoice"),
    ("billing history", "invoice"),
    ("invoice history", "invoice"),
    ("page views", "metric"),
    ("page view", "metric"),
    ("conversion funnel", "funnel"),
    ("conversion pipeline", "funnel"),
    ("conversion path", "funnel"),
    ("work item", "ticket"),
    ("crm deal", "deal"),
    ("new deal", "deal"),
    ("mockup", "file"),
    ("design file", "file"),
]

_ACTION_MAP = {
    # send actions
    "send": "send", "post": "send", "message": "send",
    "notify": "send", "tell": "send", "ping": "send", "write": "send",
    "dispatch": "send", "blast": "send", "drop": "send",
    "mail": "send", "sent": "send",
    # search actions
    "search": "search", "find": "search", "lookup": "search",
    "query": "search", "filter": "search", "scan": "search",
    "hunt": "search", "locate": "search",
    # create actions
    "create": "create", "make": "create", "new": "create",
    "open": "create", "file": "create", "book": "create",
    "schedule": "create",
    # set actions (distinct from create — for status, config)
    "set": "set", "configure": "set",
    # get/read actions
    "get": "get", "fetch": "get", "retrieve": "get", "show": "get",
    "check": "get", "view": "get", "pull": "get", "see": "get",
    "need": "get", "grab": "get", "display": "get", "present": "get",
    # list actions
    "list": "list", "enumerate": "list",
    # update actions
    "update": "update", "edit": "update", "change": "update", "modify": "update",
    "move": "update", "reschedule": "update", "adjust": "update", "tweak": "update",
    # delete actions
    "delete": "delete", "remove": "delete", "void": "delete", "nuke": "delete",
    # cancel actions (distinct from delete — for subscriptions, etc.)
    "cancel": "cancel", "terminate": "cancel", "discontinue": "cancel",
    "stop": "cancel", "end": "cancel",
    # reply actions
    "reply": "reply", "respond": "reply",
    # share actions
    "share": "share", "give": "share", "grant": "share",
    # upload actions
    "upload": "upload",
    # track/analytics actions
    "track": "track", "record": "track", "log": "track",
    "capture": "track", "register": "track", "note": "track",
    # identify actions
    "identify": "identify",
    # assign actions
    "assign": "assign", "delegate": "assign",
    # comment actions
    "comment": "comment",
    # charge/payment actions
    "charge": "charge", "bill": "charge", "invoice": "charge",
    "collect": "charge",
    # refund actions
    "refund": "refund",
    # subscribe actions
    "subscribe": "subscribe", "start": "subscribe",
    # draft actions
    "draft": "draft", "compose": "draft", "prepare": "draft",
    # dm action (maps to send but we handle target separately)
    "dm": "send",
}

_TARGET_MAP = {
    # messaging targets
    "message": "message", "notification": "message", "note": "message",
    "ping": "message",
    "dm": "dm",
    "channel": "channel", "channels": "channel", "room": "channel",
    "status": "status", "state": "status",
    # email targets
    "email": "email", "mail": "email", "inbox": "email",
    "draft": "draft", "drafts": "draft",
    "label": "label", "labels": "label",
    # crm targets
    "contact": "contact", "contacts": "contact", "record": "contact",
    "person": "contact", "individual": "contact", "entry": "contact",
    "info": "contact", "profile": "contact",
    "deal": "deal", "opportunity": "deal",
    # payment targets
    "payment": "payment", "charge": "payment",
    "refund": "refund",
    "customer": "customer", "account": "customer", "client": "customer",
    "invoice": "invoice", "invoices": "invoice", "bill": "invoice", "bills": "invoice",
    "statement": "invoice",
    "subscription": "subscription", "plan": "subscription", "membership": "subscription",
    # calendar targets
    "event": "event", "meeting": "event", "appointment": "event", "call": "event",
    "calendar": "event", "session": "event", "sync": "event",
    "time": "free_time", "slot": "free_time", "availability": "free_time",
    # file targets
    "file": "file", "files": "file", "document": "file", "doc": "file",
    "spreadsheet": "file", "sheet": "file",
    "folder": "folder", "directory": "folder", "location": "folder",
    # ticket targets
    "ticket": "ticket", "issue": "ticket", "bug": "ticket", "task": "ticket",
    "story": "ticket", "epic": "ticket", "jira": "ticket",
    "defect": "ticket",
    # analytics targets
    "metric": "metric", "metrics": "metric", "analytics": "metric",
    "pageviews": "metric", "visitors": "metric", "conversions": "metric",
    "funnel": "funnel", "conversion": "funnel", "pipeline": "funnel",
    "user": "user", "users": "user", "traits": "user",
    # analytics event targets
    "occurrence": "event", "activity": "event",
    # comment targets
    "remark": "comment", "response": "comment",
}

_DOMAIN_SIGNALS = {
    "messaging": ["slack", "dm", "channel", "#"],
    "email": ["email", "mail", "gmail", "inbox", "subject", "cc", "bcc", "draft"],
    "crm": ["crm", "contact", "deal", "pipeline", "hubspot", "salesforce", "customer record", "customer details", "contact info"],
    "payments": ["stripe", "charge", "refund", "invoice", "subscription", "payment", "cus_", "ch_", "sub_", "price_", "billing"],
    "calendar": ["calendar", "meeting", "event", "schedule", "appointment", "free time", "availability", "session", "10am", "11am", "noon"],
    "files": ["drive", "gdrive", "upload", "file", "folder", "share", "document"],
    "tickets": ["jira", "ticket", "issue", "bug", "epic", "story", "eng-", "ENG-", "sprint"],
    "analytics": ["analytics", "track", "funnel", "metric", "pageview", "conversion", "identify", "upgraded", "traits", "switched from", "changed their", "checkout", "occurrence", "activity", "button_click", "signup_completed", "page_view"],
}

_DOMAIN_SIGNAL_WEIGHTS = {
    "messaging": {"slack": 3, "dm": 2, "channel": 2, "#": 1},
    "email": {"email": 2, "mail": 2, "gmail": 3, "inbox": 2, "subject": 2, "cc": 2, "bcc": 2, "draft": 1},
    "crm": {"crm": 3, "contact": 2, "deal": 2, "pipeline": 1, "hubspot": 3, "salesforce": 3, "customer record": 3, "customer details": 3, "contact info": 3},
    "payments": {"stripe": 3, "charge": 2, "refund": 2, "invoice": 2, "subscription": 2, "payment": 2, "cus_": 3, "ch_": 3, "sub_": 2, "price_": 3, "billing": 2},
    "calendar": {"calendar": 3, "meeting": 2, "event": 1, "schedule": 2, "appointment": 2, "free time": 2, "availability": 2, "session": 1, "10am": 2, "11am": 2, "noon": 2},
    "files": {"drive": 3, "gdrive": 3, "upload": 2, "file": 1, "folder": 2, "share": 1, "document": 1},
    "tickets": {"jira": 3, "ticket": 2, "issue": 1, "bug": 2, "epic": 2, "story": 1, "eng-": 3, "ENG-": 3, "sprint": 2},
    "analytics": {"analytics": 3, "track": 2, "funnel": 3, "metric": 2, "pageview": 3, "conversion": 2, "identify": 2, "upgraded": 2, "traits": 2, "switched from": 3, "changed their": 3, "checkout": 2, "occurrence": 1, "activity": 1, "button_click": 3, "signup_completed": 3, "page_view": 3},
}

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "use", "call", "run", "execute", "tool", "function", "api",
}


def _extract_action(text: str, words: list[str]) -> str:
    text_lower = text.lower()

    # Phase 1: multi-word phrase matching for critical disambiguation
    for phrase, action in _PHRASE_ACTIONS:
        if phrase in text_lower:
            return action

    # Phase 2: single-word matching with impact ranking
    _IMPACT_RANK = {
        "charge": 10, "refund": 10, "cancel": 10, "delete": 10,
        "create": 9, "subscribe": 9,
        "update": 8, "assign": 8, "share": 8, "upload": 8,
        "identify": 7, "track": 7, "comment": 7,
        "set": 6,
        "send": 5, "draft": 5,
        "reply": 6,
        "search": 3, "get": 2, "list": 2,
    }
    _WEAK_ACTION_WORDS = {"new", "open", "file", "start", "record", "book", "move",
                          "note", "stop", "end", "drop", "present", "display",
                          "capture", "register", "collect", "prepare",
                          "mail", "sent"}
    # Words that are only weak when preceded by certain nouns
    _CONTEXTUAL_WEAK = {
        "update": {"project", "status", "the"},
    }

    found_actions = []
    strong_actions = []
    for idx, w in enumerate(words):
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _ACTION_MAP:
            action = _ACTION_MAP[clean]
            found_actions.append(action)
            is_weak = clean in _WEAK_ACTION_WORDS
            if not is_weak and clean in _CONTEXTUAL_WEAK:
                prev_word = re.sub(r"[^a-z]", "", words[idx - 1]) if idx > 0 else ""
                if prev_word in _CONTEXTUAL_WEAK[clean]:
                    is_weak = True
            if not is_weak:
                strong_actions.append(action)

    candidates = strong_actions if strong_actions else found_actions
    if candidates:
        unique_actions = list(dict.fromkeys(candidates))
        if len(unique_actions) > 1:
            return max(unique_actions, key=lambda a: _IMPACT_RANK.get(a, 0))
        return unique_actions[0]

    return "get"


def _extract_target(text: str, words: list[str], action: str = "") -> str:
    text_lower = text.lower()

    # Phase 1: multi-word phrase matching
    for phrase, target in _PHRASE_TARGETS:
        if phrase in text_lower:
            return target

    # Phase 2: channel vs DM disambiguation
    if re.search(r"#\w+", text):
        return "channel"
    if "dm " in text_lower or text_lower.startswith("dm ") or "direct message" in text_lower:
        return "dm"
    if re.search(r'\b\S+@\S+\.\S+', text) and not re.search(r"#\w+", text):
        if "message" in text_lower or "send a message" in text_lower:
            return "dm"

    # Phase 3: single-word matching
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _TARGET_MAP:
            return _TARGET_MAP[clean]

    return "general"


def _infer_domain(text: str) -> str:
    text_lower = text.lower()
    text_no_emails = re.sub(r'\S+@\S+\.\S+', '', text_lower)
    scores = {}
    _BOUNDARY_SIGNALS = {"cc", "bcc", "dm", "#", "mail", "bug"}

    for domain, signals in _DOMAIN_SIGNALS.items():
        weights = _DOMAIN_SIGNAL_WEIGHTS.get(domain, {})
        score = 0
        for s in signals:
            s_lower = s.lower()
            if s_lower == "#":
                # Special: '#' is non-word, \b doesn't apply — just check for '#' + word
                if re.search(r'#\w+', text_no_emails):
                    score += weights.get(s, 1)
            elif s_lower in _BOUNDARY_SIGNALS or len(s_lower) <= 3:
                if re.search(r'\b' + re.escape(s_lower) + r'\b', text_no_emails):
                    score += weights.get(s, 1)
            else:
                if s_lower in text_no_emails:
                    score += weights.get(s, 1)
        if score > 0:
            scores[domain] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

# Informational question patterns — these are never tool invocations.
# We suppress lexicon signals but keep BoW so the threshold handles it.
_QUESTION_PATTERNS = [
    "what does", "what is", "what are", "what's the", "whats the",
    "how does", "how do", "how many", "how much",
    "why does", "why is",
    "can you explain", "tell me about",
]

_GREETING_PATTERNS = [
    "hello", "hi there", "hey there", "how are you",
]

# Actions that are clearly outside the SaaS tool catalog.
# These are system/devops operations, not SaaS API calls.
_NON_SAAS_PATTERNS = [
    "deploy the", "deploy to", "deploy application",
    "merge the", "merge branch",
    "restart the", "restart server",
    "run the ci", "run ci", "ci pipeline", "ci/cd",
    "pull request", "database server", "dns records",
]


def _is_question(text: str) -> bool:
    """Detect if the query is an informational question, not a tool command."""
    text_lower = text.lower()
    for pattern in _QUESTION_PATTERNS:
        if pattern in text_lower:
            return True
    for pattern in _GREETING_PATTERNS:
        if text_lower.startswith(pattern):
            return True
    return False


def _is_non_saas(text: str) -> bool:
    """Detect system/devops actions that are outside the SaaS tool catalog."""
    text_lower = text.lower()
    for pattern in _NON_SAAS_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def _split_camel_case(text: str) -> str:
    """Split camelCase and PascalCase tokens into separate words.
    E.g., 'createJiraStory' → 'create Jira Story'
    Must be applied BEFORE lowercasing to detect case boundaries.
    Skips short tokens (< 4 chars) to preserve abbreviations like 'DM', 'dM'.
    """
    parts = []
    for token in text.split():
        if len(token) >= 4 and re.search(r'[a-z][A-Z]', token):
            parts.append(re.sub(r'([a-z])([A-Z])', r'\1 \2', token))
        else:
            parts.append(token)
    return " ".join(parts)


def encode_query(query: str) -> dict:
    """Convert a raw NL query into a Concept-compatible dict for tool routing.

    Returns a dict with 'name' and 'attributes' keys. Attributes include:
      action, target, domain — categorical signals from lightweight extraction
      description — full query text (stop words removed) for BoW matching
      keywords — same as description, for BoW matching against exemplar keywords

    For informational questions ("what is...", "how does..."), lexicon signals
    (action, domain) are suppressed so the threshold handles abstention.
    BoW signals are preserved for any residual matching.
    """
    # Split camelCase BEFORE lowercasing to detect case boundaries.
    # "createJiraStory" → "create Jira Story" → "create jira story"
    # Snake_case tokens like "stripe_pause_subscription" stay intact
    # (one token, not split — prevents enriching fake tool names).
    camel_split = _split_camel_case(query)
    cleaned = re.sub(r"[^\w\s#@._\-]", "", camel_split.lower())
    words = cleaned.split()

    action = _extract_action(query, words)
    target = _extract_target(query, words, action=action)
    domain = _infer_domain(query)

    # BoW text: stop words removed
    bow_text = " ".join(w for w in words if w not in _STOP_WORDS)

    # Suppress lexicon signals for informational questions/greetings
    # and for clearly non-SaaS operations (deploy, CI, merge, etc.)
    if _is_question(query) or _is_non_saas(query):
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
    # Check snake_case patterns
    for m in _SNAKE_CASE_RE.finditer(query):
        candidate = m.group(1)
        # Filter out ID prefixes
        if any(candidate.startswith(p) for p in _ID_PREFIXES):
            continue
        # Filter out event/data names: preceded by "event", "a", "an"
        # or followed by "event", "events"
        start = m.start()
        end = m.end()
        prefix = query[:start].rstrip().lower()
        suffix = query[end:].lstrip().lower()
        if prefix.endswith(("event", " a", " an")):
            continue
        if suffix.startswith(("event", "events")):
            continue
        # Split on underscore → space-separated tokens
        return candidate.replace("_", " ").lower()

    # Check camelCase patterns
    for m in _CAMEL_CASE_RE.finditer(query):
        candidate = m.group(1)
        # Split camelCase → tokens
        tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', candidate).lower()
        return tokens

    return None


def _extract_action_from_tokens(tokens: str) -> str:
    """Extract the primary action verb from space-separated tool name tokens."""
    words = tokens.split()
    for w in words:
        if w in _ACTION_MAP:
            return _ACTION_MAP[w]
    return "none"


# ---------------------------------------------------------------------------
# ToolNameSidecar — validates tool references against the real catalog
# ---------------------------------------------------------------------------

class ToolNameSidecar:
    """Sidecar model that validates tool name references in an independent
    vector space (seed=73).

    Encodes the 38 real tool names as BoW + action lexicon glyphs.
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
            exemplar_path = str(
                Path(__file__).parent / "data" / "exemplars.jsonl"
            )
        self._build_catalog(exemplar_path)

    def _build_catalog(self, exemplar_path: str):
        """Build one sidecar glyph per unique tool_id."""
        import json

        seen = set()
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

                # name_tokens: tool_id split on '_'
                name_tokens = tool_id.replace("_", " ")
                # action: from exemplar data
                action = entry.get("action", "none")

                concept = self._Concept(
                    name=f"sidecar_{tool_id}",
                    attributes={
                        "name_tokens": name_tokens,
                        "action": action,
                    },
                )
                glyph = self.encoder.encode(concept)
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

        # Encode the referenced tool name in sidecar space
        action = _extract_action_from_tokens(tool_ref_tokens)
        ref_concept = self._Concept(
            name="ref",
            attributes={
                "name_tokens": tool_ref_tokens,
                "action": action,
            },
        )
        ref_glyph = self.encoder.encode(ref_concept)

        # Role-level weighted scoring in sidecar space
        sidecar_weights = {"name_tokens": 0.7, "action": 0.3}

        ref_roles = {}
        for layer in ref_glyph.layers.values():
            for seg in layer.segments.values():
                ref_roles.update(seg.roles)

        best_score = 0.0
        best_tool = None

        for glyph, tool_id in self.tool_glyphs:
            cat_roles = {}
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

        if best_score >= threshold:
            return best_tool
        return None
