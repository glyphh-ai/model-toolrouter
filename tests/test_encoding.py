"""Test encoder config validation and basic encoding."""

from encoder import (
    ENCODER_CONFIG, SIDECAR_CONFIG,
    encode_query, entry_to_record, extract_tool_reference,
)
from glyphh.core.types import Concept
from glyphh.encoder import Encoder


def test_config_has_two_layers():
    assert len(ENCODER_CONFIG.layers) == 2
    names = [l.name for l in ENCODER_CONFIG.layers]
    assert "intent" in names
    assert "semantics" in names


def test_intent_layer_has_action_and_scope():
    intent = [l for l in ENCODER_CONFIG.layers if l.name == "intent"][0]
    seg_names = [s.name for s in intent.segments]
    assert "action" in seg_names
    assert "scope" in seg_names


def test_action_segment_has_action_and_target():
    intent = [l for l in ENCODER_CONFIG.layers if l.name == "intent"][0]
    action_seg = [s for s in intent.segments if s.name == "action"][0]
    role_names = [r.name for r in action_seg.roles]
    assert "action" in role_names
    assert "target" in role_names


def test_semantics_layer_has_bow_roles():
    semantics = [l for l in ENCODER_CONFIG.layers if l.name == "semantics"][0]
    text_seg = [s for s in semantics.segments if s.name == "text"][0]
    role_names = [r.name for r in text_seg.roles]
    assert "description" in role_names
    assert "keywords" in role_names
    for role in text_seg.roles:
        assert role.text_encoding == "bag_of_words", (
            f"Role {role.name} should use bag_of_words encoding"
        )


def test_encode_query_returns_valid_concept():
    result = encode_query("Send a message to #engineering saying the deploy is done")
    assert "name" in result
    assert "attributes" in result
    attrs = result["attributes"]
    assert attrs["action"] == "send"
    assert attrs["domain"] == "messaging"
    assert "description" in attrs
    assert "keywords" in attrs
    assert len(attrs["description"]) > 0
    assert len(attrs["keywords"]) > 0


def test_encode_query_extracts_action():
    assert encode_query("Refund charge ch_abc")["attributes"]["action"] == "refund"
    assert encode_query("Create a new ticket")["attributes"]["action"] == "create"
    assert encode_query("Search for files")["attributes"]["action"] == "search"
    assert encode_query("Cancel the subscription")["attributes"]["action"] == "cancel"


def test_encode_query_extracts_domain():
    assert encode_query("Send a Slack message")["attributes"]["domain"] == "messaging"
    assert encode_query("Send an email to bob")["attributes"]["domain"] == "email"
    assert encode_query("Look up the CRM contact")["attributes"]["domain"] == "crm"
    assert encode_query("Charge the Stripe customer")["attributes"]["domain"] == "payments"
    assert encode_query("Create a Jira ticket")["attributes"]["domain"] == "tickets"


def test_encode_query_description_is_bow_text():
    result = encode_query("Refund charge ch_abc for the full amount")
    desc = result["attributes"]["description"]
    # Stop words removed, meaningful words preserved
    assert "refund" in desc
    assert "charge" in desc
    assert "the" not in desc.split()  # stop word removed


def test_encode_query_suppresses_questions():
    """Informational questions suppress lexicon signals (action/domain) so the
    threshold can reject them via low BoW similarity, not hardcoded patterns."""
    result = encode_query("What's the weather like in San Francisco?")
    attrs = result["attributes"]
    # Questions get action="none" and domain="none" to suppress lexicon matching,
    # but BoW (description/keywords) is preserved so threshold handles abstention.
    assert attrs["action"] == "none"
    assert attrs["domain"] in ("none", "general")
    assert len(attrs["description"]) > 0  # BoW text preserved


def test_entry_to_record_structure():
    entry = {
        "tool_id": "slack_send_message",
        "action": "send",
        "target": "channel",
        "domain": "messaging",
        "keywords": ["send", "message", "slack", "channel"],
        "description": "send message slack channel post notify",
    }
    record = entry_to_record(entry)
    assert "concept_text" in record
    assert "attributes" in record
    assert "metadata" in record
    assert record["metadata"]["tool_id"] == "slack_send_message"
    assert record["attributes"]["action"] == "send"
    assert record["attributes"]["description"] == "send message slack channel post notify"
    assert record["attributes"]["keywords"] == "send message slack channel"


def test_entry_to_record_joins_keyword_list():
    entry = {
        "tool_id": "test_tool",
        "action": "get",
        "target": "contact",
        "domain": "crm",
        "keywords": ["get", "contact", "crm"],
        "description": "get crm contact",
    }
    record = entry_to_record(entry)
    assert record["attributes"]["keywords"] == "get contact crm"


def test_encoder_produces_glyph(encoder):
    concept = Concept(
        name="test_concept",
        attributes={
            "action": "send",
            "target": "channel",
            "domain": "messaging",
            "description": "send message slack channel",
            "keywords": "send message slack channel",
        },
    )
    glyph = encoder.encode(concept)
    assert glyph is not None
    assert "intent" in glyph.layers
    assert "semantics" in glyph.layers


# --- Sidecar config tests ---

def test_sidecar_config_has_two_layers():
    assert len(SIDECAR_CONFIG.layers) == 2
    names = [l.name for l in SIDECAR_CONFIG.layers]
    assert "identity" in names
    assert "capability" in names


def test_sidecar_uses_independent_seed():
    assert SIDECAR_CONFIG.seed == 73
    assert ENCODER_CONFIG.seed == 42
    assert SIDECAR_CONFIG.seed != ENCODER_CONFIG.seed


# --- Tool reference extraction tests ---

def test_extract_tool_ref_snake_case():
    assert extract_tool_reference("Use stripe_pause_subscription to pause") == "stripe pause subscription"


def test_extract_tool_ref_camel_case():
    assert extract_tool_reference("Call sendSlackNotification to tell #ops") == "send slack notification"


def test_extract_tool_ref_filters_ids():
    assert extract_tool_reference("Refund charge ch_3abc123") is None
    assert extract_tool_reference("Cancel sub_xyz789") is None


def test_extract_tool_ref_filters_event_names():
    assert extract_tool_reference("Track a signup_completed event") is None
    assert extract_tool_reference("Track the button_click event on pricing") is None


def test_extract_tool_ref_no_match():
    assert extract_tool_reference("Send a message to #engineering") is None


# --- Sidecar validation tests ---

def test_sidecar_validates_real_tool(sidecar):
    result = sidecar.validate("slack send message")
    assert result == "slack_send_message"


def test_sidecar_rejects_fake_tool(sidecar):
    result = sidecar.validate("stripe pause subscription")
    assert result is None


def test_sidecar_maps_synonym_to_real_tool(sidecar):
    # camelCase "sendSlackNotification" → "send slack notification"
    result = sidecar.validate("send slack notification")
    assert result == "slack_send_message"
