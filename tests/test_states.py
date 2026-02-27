"""Tests for assess_query() — slot completeness and disambiguation logic.

These tests run entirely offline (no runtime, no server).  They verify that
the model's encoder correctly identifies complete vs incomplete queries so the
runtime can decide between DONE and ASK states.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import assess_query


# ---------------------------------------------------------------------------
# Complete queries — should not trigger ASK
# ---------------------------------------------------------------------------

def test_slack_send_is_complete():
    result = assess_query("send a message to the team on Slack")
    assert result["complete"], f"Expected complete, got missing={result['missing']}"
    assert result["action"] == "send"
    assert result["domain"] == "messaging"
    assert result["missing"] == []


def test_jira_create_is_complete():
    result = assess_query("create a new Jira ticket for the login bug")
    assert result["complete"], f"Expected complete, got missing={result['missing']}"
    assert result["domain"] == "tickets"


def test_stripe_refund_is_complete():
    result = assess_query("refund the last charge on Stripe")
    assert result["complete"], f"Expected complete, got missing={result['missing']}"
    assert result["domain"] == "payments"


def test_gdrive_search_is_complete():
    result = assess_query("find the Q3 report in Google Drive")
    assert result["complete"], f"Expected complete, got missing={result['missing']}"
    assert result["domain"] == "files"


def test_jira_summary_with_ticket_ref_is_complete():
    """Real query from user — 'describe ISS-3275 in Jira' must be complete."""
    result = assess_query("hey AI, can you describe ISS-3275 to me in Jira?")
    assert result["complete"], f"Expected complete, got missing={result['missing']}"
    assert result["domain"] == "tickets"


# ---------------------------------------------------------------------------
# Incomplete queries — should trigger ASK
# ---------------------------------------------------------------------------

def test_trailing_incomplete_query_triggers_ask():
    """Trailing off with '?' — no domain, no action."""
    result = assess_query("hey AI, can you describe ISS-3275 to me in ?")
    assert not result["complete"]
    assert len(result["missing"]) > 0
    assert result["reason"] != ""


def test_generic_do_something_triggers_ask():
    result = assess_query("do something")
    assert not result["complete"]
    assert "action" in result["missing"] or "domain" in result["missing"]


def test_greeting_triggers_ask():
    result = assess_query("hello")
    assert not result["complete"]


def test_vague_help_triggers_ask():
    result = assess_query("help me with this")
    assert not result["complete"]


def test_missing_domain_is_reported():
    """Action present but domain unresolvable → missing should contain 'domain'."""
    result = assess_query("send this")
    if not result["complete"]:
        assert "domain" in result["missing"] or "action" in result["missing"]


# ---------------------------------------------------------------------------
# Return shape contract
# ---------------------------------------------------------------------------

def test_assess_query_returns_required_keys():
    result = assess_query("send a slack message")
    for key in ("complete", "missing", "reason", "action", "domain"):
        assert key in result, f"Missing key: {key}"


def test_complete_query_has_empty_reason():
    result = assess_query("send a Slack message to #general")
    if result["complete"]:
        assert result["reason"] == ""


def test_incomplete_query_has_nonempty_reason():
    result = assess_query("do something with it")
    if not result["complete"]:
        assert result["reason"] != ""
