"""Test encoder config validation and basic encoding."""

from encoder import ENCODER_CONFIG, encode_query, entry_to_record
from glyphh.core.types import Concept
from glyphh.encoder import Encoder


def test_config_has_two_layers():
    assert len(ENCODER_CONFIG.layers) == 2
    names = [l.name for l in ENCODER_CONFIG.layers]
    assert "intent" in names
    assert "context" in names


def test_intent_layer_has_action_and_scope():
    intent = [l for l in ENCODER_CONFIG.layers if l.name == "intent"][0]
    seg_names = [s.name for s in intent.segments]
    assert "action" in seg_names
    assert "scope" in seg_names


def test_action_segment_has_verb_and_object():
    intent = [l for l in ENCODER_CONFIG.layers if l.name == "intent"][0]
    action = [s for s in intent.segments if s.name == "action"][0]
    role_names = [r.name for r in action.roles]
    assert "verb" in role_names
    assert "object" in role_names


def test_encode_query_returns_valid_concept():
    result = encode_query("create a release branch")
    assert "name" in result
    assert "attributes" in result
    attrs = result["attributes"]
    assert attrs["verb"] == "build"
    assert attrs["object"] == "branch"
    assert attrs["domain"] == "release"


def test_encode_query_docker_domain():
    result = encode_query("rebuild the docker containers")
    assert result["attributes"]["domain"] == "docker"


def test_encode_query_test_domain():
    result = encode_query("run the pytest suite")
    assert result["attributes"]["domain"] == "test"


def test_entry_to_record_structure():
    entry = {
        "question": "tag the release",
        "tool_id": "create_tag",
        "verb": "deploy",
        "object": "tag",
        "domain": "release",
        "keywords": ["tag", "release"],
        "description": "Tag the release",
    }
    record = entry_to_record(entry)
    assert "concept_text" in record
    assert "attributes" in record
    assert "metadata" in record
    assert record["metadata"]["tool_id"] == "create_tag"
    assert record["attributes"]["verb"] == "deploy"


def test_encoder_produces_glyph(encoder):
    concept = Concept(
        name="test_concept",
        attributes={"verb": "build", "object": "branch", "domain": "release", "keywords": "test"},
    )
    glyph = encoder.encode(concept)
    assert glyph is not None
    assert "intent" in glyph.layers
