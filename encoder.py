"""
Custom encoder for the SaaS tool router model.

Exports:
  ENCODER_CONFIG — EncoderConfig with intent-focused routing layers
  encode_query(query) — converts NL text to a Concept dict for similarity search

Uses Glyphh HDC primitives:
  - Two-layer architecture: intent (action/target/domain) + context (keywords)
  - Action carries highest weight for verb disambiguation
  - Domain layer separates messaging/email/crm/payments/calendar/files/tickets/analytics
  - Lexicons on roles for NL query matching
"""

import hashlib
import json
import re
from pathlib import Path

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
                            similarity_weight=0.8,
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
                            similarity_weight=1.0,
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
                                "send", "message", "email", "search", "find",
                                "create", "update", "delete", "refund", "charge",
                                "cancel", "subscribe", "schedule", "meeting",
                                "ticket", "bug", "file", "upload", "share",
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

# Multi-word phrases that should be matched BEFORE single-word action extraction.
# Order matters: longer/more-specific phrases first.
_PHRASE_ACTIONS = [
    # comment-specific (must beat "add" → create)
    ("add a comment", "comment"),
    ("add comment", "comment"),
    ("leave a comment", "comment"),
    ("leave comment", "comment"),
    ("post a comment", "comment"),
    # status-specific (must beat "set" → create)
    ("set status", "set"),
    ("set my status", "set"),
    ("set my slack status", "set"),
    # share/access
    ("give access", "share"),
    ("grant access", "share"),
    ("read-only access", "share"),
    ("read only access", "share"),
    # assign-specific (must beat "update" when "assign" is in the query)
    ("assign it to", "assign"),
    ("assign to", "assign"),
    ("assign ticket", "assign"),
    # list-specific (must beat "look up" → search)
    ("look up what channels", "list"),
    ("look up channels", "list"),
    # search-specific
    ("search for the invoice email", "search"),
    ("search for the email", "search"),
    # get-specific
    ("look up", "get"),
    ("look for", "search"),
    ("find the customer", "get"),
    ("find the contact", "get"),
    ("find the record", "get"),
    # identify (trait changes) — must come before track/log
    ("upgraded to", "identify"),
    ("downgraded to", "identify"),
    ("changed plan", "identify"),
    # track/log
    ("log that", "track"),
    ("log event", "track"),
    # dm-specific
    ("send a dm", "send"),
    ("send dm", "send"),
    # payment history
    ("payment history", "list"),
    ("billing history", "list"),
    ("invoice history", "list"),
]

# Multi-word phrases for target extraction
_PHRASE_TARGETS = [
    ("invoice email", "email"),
    ("payment history", "invoice"),
    ("billing history", "invoice"),
    ("invoice history", "invoice"),
    ("free time", "free_time"),
    ("available time", "free_time"),
    ("time slot", "free_time"),
    ("add a comment", "comment"),
    ("add comment", "comment"),
    ("leave a comment", "comment"),
    ("leave comment", "comment"),
    ("post a comment", "comment"),
    ("set status", "status"),
    ("slack status", "status"),
    ("my status", "status"),
    ("direct message", "dm"),
    ("page views", "metric"),
    ("page_views", "metric"),
    ("pageviews", "metric"),
    ("upgraded to", "user"),
    ("downgraded to", "user"),
]

_ACTION_MAP = {
    # send actions
    "send": "send", "post": "send", "message": "send",
    "notify": "send", "tell": "send", "ping": "send", "write": "send",
    # search actions
    "search": "search", "find": "search", "lookup": "search",
    "query": "search", "filter": "search",
    # create actions
    "create": "create", "make": "create", "new": "create",
    "open": "create", "file": "create", "book": "create",
    "schedule": "create",
    # set actions (distinct from create — for status, config)
    "set": "set",
    # get/read actions
    "get": "get", "fetch": "get", "retrieve": "get", "show": "get",
    "check": "get", "view": "get", "pull": "get", "see": "get",
    # list actions
    "list": "list", "enumerate": "list",
    # update actions
    "update": "update", "edit": "update", "change": "update", "modify": "update",
    "move": "update", "reschedule": "update",
    # delete actions
    "delete": "delete", "remove": "delete", "void": "delete",
    # cancel actions (distinct from delete — for subscriptions, etc.)
    "cancel": "cancel",
    # reply actions
    "reply": "reply", "respond": "reply",
    # share actions
    "share": "share", "give": "share", "grant": "share",
    # upload actions
    "upload": "upload",
    # track/analytics actions
    "track": "track", "record": "track", "log": "track",
    # identify actions
    "identify": "identify",
    # assign actions
    "assign": "assign",
    # comment actions
    "comment": "comment",
    # charge/payment actions
    "charge": "charge", "bill": "charge", "invoice": "charge",
    # refund actions
    "refund": "refund",
    # subscribe actions
    "subscribe": "subscribe", "start": "subscribe",
    # draft actions
    "draft": "draft",
    # dm action (maps to send but we handle target separately)
    "dm": "send",
}

_TARGET_MAP = {
    # messaging targets
    "message": "message", "notification": "message",
    "dm": "dm",
    "channel": "channel", "channels": "channel",
    "status": "status",
    # email targets
    "email": "email", "mail": "email", "inbox": "email",
    "draft": "draft", "drafts": "draft",
    "label": "label", "labels": "label",
    # crm targets
    "contact": "contact", "contacts": "contact", "record": "contact",
    "deal": "deal", "opportunity": "deal",
    # payment targets
    "payment": "payment", "charge": "payment",
    "refund": "refund",
    "customer": "customer", "account": "customer",
    "invoice": "invoice", "invoices": "invoice", "bill": "invoice", "bills": "invoice",
    "subscription": "subscription", "plan": "subscription",
    # calendar targets
    "event": "event", "meeting": "event", "appointment": "event", "call": "event",
    "calendar": "event",
    "time": "free_time", "slot": "free_time", "availability": "free_time",
    # file targets
    "file": "file", "files": "file", "document": "file", "doc": "file",
    "spreadsheet": "file", "sheet": "file",
    "folder": "folder",
    # ticket targets
    "ticket": "ticket", "issue": "ticket", "bug": "ticket", "task": "ticket",
    "story": "ticket", "epic": "ticket", "jira": "ticket",
    # analytics targets
    "metric": "metric", "metrics": "metric", "analytics": "metric",
    "pageviews": "metric", "visitors": "metric", "conversions": "metric",
    "funnel": "funnel", "conversion": "funnel",
    "user": "user", "users": "user", "traits": "user",
}

_DOMAIN_SIGNALS = {
    "messaging": ["slack", "dm", "channel", "#"],
    "email": ["email", "mail", "gmail", "inbox", "subject", "cc", "bcc", "draft"],
    "crm": ["crm", "contact", "deal", "pipeline", "hubspot", "salesforce", "customer record"],
    "payments": ["stripe", "charge", "refund", "invoice", "subscription", "payment", "cus_", "ch_", "sub_", "price_"],
    "calendar": ["calendar", "meeting", "event", "schedule", "appointment", "free time", "availability"],
    "files": ["drive", "gdrive", "upload", "file", "folder", "share", "document"],
    "tickets": ["jira", "ticket", "issue", "bug", "epic", "story", "eng-", "ENG-", "sprint"],
    "analytics": ["analytics", "track", "funnel", "metric", "pageview", "conversion", "identify", "upgraded", "traits"],
}

# Weighted domain signals — some signals are stronger indicators than others
_DOMAIN_SIGNAL_WEIGHTS = {
    "messaging": {"slack": 3, "dm": 2, "channel": 2, "#": 2},
    "email": {"email": 2, "mail": 2, "gmail": 3, "inbox": 2, "subject": 2, "cc": 2, "bcc": 2, "draft": 1},
    "crm": {"crm": 3, "contact": 2, "deal": 2, "pipeline": 1, "hubspot": 3, "salesforce": 3, "customer record": 3},
    "payments": {"stripe": 3, "charge": 2, "refund": 2, "invoice": 2, "subscription": 2, "payment": 2, "cus_": 3, "ch_": 3, "sub_": 2, "price_": 3},
    "calendar": {"calendar": 3, "meeting": 2, "event": 1, "schedule": 2, "appointment": 2, "free time": 2, "availability": 2},
    "files": {"drive": 3, "gdrive": 3, "upload": 2, "file": 1, "folder": 2, "share": 1, "document": 1},
    "tickets": {"jira": 3, "ticket": 2, "issue": 1, "bug": 2, "epic": 2, "story": 1, "eng-": 3, "ENG-": 3, "sprint": 2},
    "analytics": {"analytics": 3, "track": 2, "funnel": 3, "metric": 2, "pageview": 3, "conversion": 2, "identify": 2, "upgraded": 2, "traits": 2},
}

# Out-of-scope qualifiers — if the query contains these patterns, it's likely
# NOT a tool invocation even if domain words appear.
_OOS_QUESTION_PATTERNS = [
    # Questions about policy/docs (not actions)
    "what does", "what is", "what are", "how does", "how do", "how many",
    "how much", "why does", "why is", "can you explain", "tell me about",
    "what's the", "whats the",
    # Greetings
    "hello", "hi there", "hey there", "how are you",
]

# These are OOS only when they appear as the PRIMARY action (first few words)
_OOS_ACTION_PATTERNS = [
    "deploy the", "deploy to", "deploy application",
    "merge the", "merge branch",
    "restart the", "restart server",
    "approve the", "approve pending",
    "translate this", "translate the",
    "summarize the", "summarize last",
    "generate a pdf", "generate pdf",
    "book a flight",
    "update the dns", "update dns records",
    "run the ci", "run ci",
]

# These are OOS regardless of position
_OOS_ANYWHERE_PATTERNS = [
    "text message to +", "sms to +",
    "ci pipeline", "ci/cd",
    "pull request from",
    "database server",
    "dns records for",
    "flight to",
]

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "need", "want", "think",
    "use", "call", "run", "execute", "tool", "function", "api",
}


def _is_oos_query(text: str) -> bool:
    """Check if the query matches out-of-scope patterns."""
    text_lower = text.lower()

    # Question patterns — match anywhere
    for pattern in _OOS_QUESTION_PATTERNS:
        if pattern in text_lower:
            return True

    # Anywhere patterns — match anywhere
    for pattern in _OOS_ANYWHERE_PATTERNS:
        if pattern in text_lower:
            return True

    # Action patterns — only match if they appear near the start of the query
    # (first ~40 chars) to avoid matching "deploy" in message bodies
    prefix = text_lower[:50]
    for pattern in _OOS_ACTION_PATTERNS:
        if pattern in prefix:
            return True

    return False


def _extract_action(text: str, words: list[str]) -> str:
    text_lower = text.lower()

    # Phase 1: multi-word phrase matching (highest priority)
    for phrase, action in _PHRASE_ACTIONS:
        if phrase in text_lower:
            return action

    # Phase 2: single-word matching
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _ACTION_MAP:
            return _ACTION_MAP[clean]

    # Phase 3: extract action from camelCase tool names in the query
    # e.g. "createJiraStory" → "create", "sendSlackNotification" → "send"
    camel_pattern = re.compile(r'([a-z]+)[A-Z]')
    for w in text.split():
        m = camel_pattern.match(w)
        if m:
            verb = m.group(1).lower()
            if verb in _ACTION_MAP:
                return _ACTION_MAP[verb]

    return "get"


def _extract_target(text: str, words: list[str]) -> str:
    text_lower = text.lower()

    # Phase 1: multi-word phrase matching
    for phrase, target in _PHRASE_TARGETS:
        if phrase in text_lower:
            return target

    # Phase 2: channel vs DM disambiguation
    # If we see # followed by a word, it's a channel message
    if re.search(r"#\w+", text):
        return "channel"
    # If "dm" or "direct message" appears, it's a DM
    if "dm " in text_lower or text_lower.startswith("dm ") or "direct message" in text_lower:
        return "dm"
    # If an email address is present and the action is "send"/"message",
    # it's likely a DM (not a channel message)
    if re.search(r'\b\S+@\S+\.\S+', text) and not re.search(r"#\w+", text):
        # Check if the context suggests messaging (not email)
        if "message" in text_lower or "send a message" in text_lower:
            return "dm"

    # Phase 3: single-word matching
    for w in words:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _TARGET_MAP:
            return _TARGET_MAP[clean]

    # Phase 4: extract target from camelCase tool names
    # e.g. "createJiraStory" → look for "story" → "ticket"
    camel_parts = re.compile(r'[A-Z][a-z]+')
    for w in text.split():
        parts = camel_parts.findall(w)
        for part in parts:
            part_lower = part.lower()
            if part_lower in _TARGET_MAP:
                return _TARGET_MAP[part_lower]

    return "general"


def _infer_domain(text: str) -> str:
    text_lower = text.lower()
    # Strip email addresses to avoid false domain signals from "@" addresses
    text_no_emails = re.sub(r'\S+@\S+\.\S+', '', text_lower)
    scores = {}
    for domain, signals in _DOMAIN_SIGNALS.items():
        weights = _DOMAIN_SIGNAL_WEIGHTS.get(domain, {})
        score = 0
        for s in signals:
            s_lower = s.lower()
            # For short signals (<=3 chars), use word boundary matching
            # to avoid false positives like "cc" in "access"
            if len(s_lower) <= 3:
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


# Known real tool names in the catalog
_REAL_TOOLS = {
    "slack_send_message", "slack_send_dm", "slack_set_status",
    "slack_list_channels", "slack_search_messages",
    "email_send", "email_search", "email_reply",
    "email_create_draft", "email_list_labels",
    "crm_get_contact", "crm_create_contact", "crm_update_contact",
    "crm_search_contacts", "crm_create_deal",
    "stripe_charge", "stripe_refund", "stripe_get_customer",
    "stripe_list_invoices", "stripe_create_subscription",
    "stripe_cancel_subscription",
    "calendar_create_event", "calendar_list_events",
    "calendar_delete_event", "calendar_find_free_time",
    "gdrive_upload", "gdrive_search", "gdrive_share",
    "gdrive_create_folder",
    "jira_create_ticket", "jira_update_ticket", "jira_search",
    "jira_add_comment", "jira_assign_ticket",
    "analytics_track_event", "analytics_get_metrics",
    "analytics_get_funnel", "analytics_identify_user",
}

# Patterns that look like tool/function names (snake_case identifiers)
_TOOL_NAME_PATTERN = re.compile(r'\b([a-z]+_[a-z_]+[a-z])\b')

# Phrases that indicate the user is explicitly naming a tool
_TOOL_INVOCATION_CUES = [
    "use the ", "use ", "call the ", "call ", "run the ", "run ",
    "execute the ", "execute ", "invoke the ", "invoke ",
    "tool is called ", "tool called ",
]


def _mentions_fake_tool(text: str) -> bool:
    """Detect if the user explicitly names a tool that doesn't exist,
    AND the underlying intent doesn't map to a real tool action."""
    text_lower = text.lower()

    # Only check if there's a tool invocation cue
    has_cue = any(cue in text_lower for cue in _TOOL_INVOCATION_CUES)
    if not has_cue:
        return False

    # Find all snake_case identifiers that look like tool names
    candidates = _TOOL_NAME_PATTERN.findall(text_lower)
    fake_found = False
    for candidate in candidates:
        if candidate not in _REAL_TOOLS and len(candidate) > 5:
            fake_found = True
            break

    if not fake_found:
        # Also check for camelCase tool names (e.g. "sendSlackNotification")
        camel_pattern = re.compile(r'\b([a-z]+[A-Z][a-zA-Z]+)\b')
        camel_candidates = camel_pattern.findall(text)
        for candidate in camel_candidates:
            snake = re.sub(r'([A-Z])', r'_\1', candidate).lower()
            if snake not in _REAL_TOOLS:
                fake_found = True
                break

    if not fake_found:
        return False

    # A fake tool name was found. But does the query's underlying intent
    # still map to a real action? If so, let it through (the user just
    # got the tool name wrong but the intent is valid).
    # We check: does the query contain strong action verbs that map to
    # real tool actions?
    _STRONG_INTENT_VERBS = {
        "refund", "charge", "send", "create", "search", "find",
        "get", "list", "update", "delete", "reply", "share",
        "upload", "track", "assign", "cancel", "tell", "post",
        "dm", "add", "set", "draft", "book", "schedule",
        "comment", "notify",
    }
    words = set(re.sub(r"[^a-z\s]", "", text_lower).split())
    if words & _STRONG_INTENT_VERBS:
        return False  # Strong intent verb present — let it through

    # No strong intent verb — the fake tool name IS the only signal.
    # Block it.
    return True


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a raw NL query into a Concept-compatible dict for tool routing."""
    cleaned = re.sub(r"[^\w\s#@._\-]", "", query.lower())
    words = cleaned.split()

    action = _extract_action(query, words)
    target = _extract_target(query, words)
    domain = _infer_domain(query)
    keywords = " ".join(w for w in words if w not in _STOP_WORDS)

    # OOS check — if the query looks out-of-scope, force domain to "general"
    # so it won't match any tool domain strongly
    if _is_oos_query(query):
        domain = "none"
        action = "none"

    # Fake tool name detection — if the user explicitly names a tool-like
    # identifier (e.g. "use stripe_pause_subscription") that doesn't exist
    # in our catalog, suppress the match
    if _mentions_fake_tool(query):
        domain = "none"
        action = "none"

    stable_id = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    return {
        "name": f"query_{stable_id:08d}",
        "attributes": {
            "action": action,
            "target": target,
            "domain": domain,
            "keywords": keywords,
        },
    }
