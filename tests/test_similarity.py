"""Test similarity-based tool routing correctness."""

from encoder import encode_query
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept


def _route(query: str, encoder, exemplar_glyphs, threshold=0.35):
    """Route a query and return (tool_id, confidence)."""
    q_concept = Concept(name="q", attributes=encode_query(query)["attributes"])
    q_glyph = encoder.encode(q_concept)

    q_intent = q_glyph.layers.get("intent") and q_glyph.layers["intent"].segments.get("action")

    best_score, best_tool = 0.0, None
    for eg, entry in exemplar_glyphs:
        e_intent = eg.layers.get("intent") and eg.layers["intent"].segments.get("action")
        if q_intent and e_intent:
            score = float(cosine_similarity(q_intent.cortex.data, e_intent.cortex.data))
        else:
            score = float(cosine_similarity(q_glyph.global_cortex.data, eg.global_cortex.data))
        if score > best_score:
            best_score = score
            best_tool = entry["tool_id"]

    if best_score >= threshold:
        return best_tool, best_score
    return None, best_score


def test_route_create_branch(encoder, exemplar_glyphs):
    tool, conf = _route("create a release branch", encoder, exemplar_glyphs)
    assert tool == "create_branch", f"Expected create_branch, got {tool} ({conf:.3f})"


def test_route_run_tests(encoder, exemplar_glyphs):
    tool, conf = _route("run the tests", encoder, exemplar_glyphs)
    assert tool == "run_tests", f"Expected run_tests, got {tool} ({conf:.3f})"


def test_route_merge(encoder, exemplar_glyphs):
    tool, conf = _route("merge to main", encoder, exemplar_glyphs)
    assert tool == "merge_to_main", f"Expected merge_to_main, got {tool} ({conf:.3f})"


def test_route_tag(encoder, exemplar_glyphs):
    tool, conf = _route("tag the release", encoder, exemplar_glyphs)
    assert tool == "create_tag", f"Expected create_tag, got {tool} ({conf:.3f})"


def test_route_check_build(encoder, exemplar_glyphs):
    tool, conf = _route("check the build status", encoder, exemplar_glyphs)
    assert tool == "check_workflow_status", f"Expected check_workflow_status, got {tool} ({conf:.3f})"


def test_route_cleanup(encoder, exemplar_glyphs):
    tool, conf = _route("delete the release branch", encoder, exemplar_glyphs)
    assert tool == "cleanup_branch", f"Expected cleanup_branch, got {tool} ({conf:.3f})"


def test_route_release_sdk(encoder, exemplar_glyphs):
    tool, conf = _route("ship the SDK", encoder, exemplar_glyphs)
    assert tool == "release_sdk", f"Expected release_sdk, got {tool} ({conf:.3f})"


def test_route_docker_rebuild(encoder, exemplar_glyphs):
    tool, conf = _route("rebuild all docker containers", encoder, exemplar_glyphs)
    assert tool == "rebuild_all", f"Expected rebuild_all, got {tool} ({conf:.3f})"


def test_route_docker_logs(encoder, exemplar_glyphs):
    tool, conf = _route("show the docker logs", encoder, exemplar_glyphs)
    assert tool == "docker_logs", f"Expected docker_logs, got {tool} ({conf:.3f})"


def test_all_test_queries(encoder, exemplar_glyphs, test_queries):
    """Run all test queries from test-queries.json and check routing accuracy."""
    correct = 0
    failures = []
    for tq in test_queries:
        tool, conf = _route(tq["query"], encoder, exemplar_glyphs)
        if tool == tq["expected_tool"]:
            correct += 1
        else:
            failures.append(f"  '{tq['query']}': expected {tq['expected_tool']}, got {tool} ({conf:.3f})")

    accuracy = correct / len(test_queries) if test_queries else 0
    if failures:
        fail_str = "\n".join(failures)
        assert accuracy >= 0.80, (
            f"Routing accuracy {accuracy:.1%} below 80% threshold.\n"
            f"Failures:\n{fail_str}"
        )
