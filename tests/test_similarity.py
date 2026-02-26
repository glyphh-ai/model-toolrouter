"""Test similarity-based tool routing correctness."""

from encoder import encode_query, extract_tool_reference
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept


ROLE_WEIGHTS = {"action": 1.0, "target": 0.7, "domain": 0.8, "description": 1.0, "keywords": 0.8}
THRESHOLD = 0.40
SIDECAR_UPPER = 0.55


def _route(query: str, encoder, exemplar_glyphs, sidecar=None, threshold=THRESHOLD):
    """Route a query using weighted role-level similarity + optional sidecar."""
    q_attrs = encode_query(query)["attributes"]
    q_concept = Concept(name="q", attributes=q_attrs)
    q_glyph = encoder.encode(q_concept)

    # Extract query role vectors
    q_roles = {}
    for layer in q_glyph.layers.values():
        for seg in layer.segments.values():
            q_roles.update(seg.roles)

    best_score, best_tool = 0.0, None
    for eg, entry in exemplar_glyphs:
        e_roles = {}
        for layer in eg.layers.values():
            for seg in layer.segments.values():
                e_roles.update(seg.roles)

        weighted_sum = 0.0
        weight_total = 0.0
        for rname, w in ROLE_WEIGHTS.items():
            if rname in q_roles and rname in e_roles:
                sim = float(cosine_similarity(q_roles[rname].data, e_roles[rname].data))
                weighted_sum += sim * w
                weight_total += w

        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        if score > best_score:
            best_score = score
            best_tool = entry["tool_id"]

    if best_score >= threshold:
        # Sidecar validation in uncertain zone
        if sidecar is not None and best_score < SIDECAR_UPPER:
            tool_ref = extract_tool_reference(query)
            if tool_ref is not None:
                sidecar_match = sidecar.validate(tool_ref)
                if sidecar_match is None:
                    return None, best_score
                else:
                    best_tool = sidecar_match
        return best_tool, best_score
    return None, best_score


# --- Clear routing tests ---

def test_route_slack_message(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Send a message to #engineering saying the deploy is done", encoder, exemplar_glyphs, sidecar)
    assert tool == "slack_send_message", f"Expected slack_send_message, got {tool} ({conf:.3f})"


def test_route_email_send(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Email the quarterly report to finance@acme.com with subject Q4 Results", encoder, exemplar_glyphs, sidecar)
    assert tool == "email_send", f"Expected email_send, got {tool} ({conf:.3f})"


def test_route_stripe_refund(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Refund charge ch_3abc123 for the full amount", encoder, exemplar_glyphs, sidecar)
    assert tool == "stripe_refund", f"Expected stripe_refund, got {tool} ({conf:.3f})"


def test_route_jira_create(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Create a bug ticket in the ENG project: login page crashes on Safari", encoder, exemplar_glyphs, sidecar)
    assert tool == "jira_create_ticket", f"Expected jira_create_ticket, got {tool} ({conf:.3f})"


def test_route_analytics_track(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Track a signup_completed event for user u_12345", encoder, exemplar_glyphs, sidecar)
    assert tool == "analytics_track_event", f"Expected analytics_track_event, got {tool} ({conf:.3f})"


def test_route_calendar_create(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Create a meeting called Sprint Planning on Monday at 10am", encoder, exemplar_glyphs, sidecar)
    assert tool == "calendar_create_event", f"Expected calendar_create_event, got {tool} ({conf:.3f})"


# --- Disambiguation tests ---

def test_disambiguate_slack_dm_vs_email(encoder, exemplar_glyphs, sidecar):
    tool, _ = _route("Send a message to alice@acme.com about the project update", encoder, exemplar_glyphs, sidecar)
    assert tool == "slack_send_dm", f"Expected slack_send_dm, got {tool}"


def test_disambiguate_crm_vs_stripe_customer(encoder, exemplar_glyphs, sidecar):
    tool, _ = _route("Find the customer record for bob@startup.io", encoder, exemplar_glyphs, sidecar)
    assert tool == "crm_get_contact", f"Expected crm_get_contact, got {tool}"


def test_disambiguate_identify_vs_track(encoder, exemplar_glyphs, sidecar):
    tool, _ = _route("Log that user u_555 just upgraded to the pro plan", encoder, exemplar_glyphs, sidecar)
    assert tool == "analytics_identify_user", f"Expected analytics_identify_user, got {tool}"


def test_disambiguate_track_event(encoder, exemplar_glyphs, sidecar):
    tool, _ = _route("Record a checkout_completed event for user u_900", encoder, exemplar_glyphs, sidecar)
    assert tool == "analytics_track_event", f"Expected analytics_track_event, got {tool}"


def test_disambiguate_comment_vs_create(encoder, exemplar_glyphs, sidecar):
    tool, _ = _route("Add a comment on ENG-4521 saying the fix is deployed to production", encoder, exemplar_glyphs, sidecar)
    assert tool == "jira_add_comment", f"Expected jira_add_comment, got {tool}"


# --- OOS / threshold-based abstention ---

def test_oos_weather(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("What's the weather like in San Francisco today?", encoder, exemplar_glyphs, sidecar)
    assert tool is None, f"Expected None for OOS query, got {tool} ({conf:.3f})"


def test_oos_greeting(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Hello, how are you?", encoder, exemplar_glyphs, sidecar)
    assert tool is None, f"Expected None for greeting, got {tool} ({conf:.3f})"


def test_oos_deploy(encoder, exemplar_glyphs, sidecar):
    tool, conf = _route("Deploy the application to production", encoder, exemplar_glyphs, sidecar)
    assert tool is None, f"Expected None for OOS deploy query, got {tool} ({conf:.3f})"


# --- Adversarial tests (sidecar validates tool references) ---

def test_adversarial_fake_tool_rejected(encoder, exemplar_glyphs, sidecar):
    """Queries referencing fake tool names should be rejected by the sidecar."""
    tool, _ = _route("Use stripe_pause_subscription to pause sub_abc for 30 days", encoder, exemplar_glyphs, sidecar)
    assert tool is None, f"Expected None for fake tool, got {tool}"


def test_adversarial_real_tool_synonym(encoder, exemplar_glyphs, sidecar):
    """Queries using camelCase synonyms should route to the real tool."""
    tool, _ = _route("Call sendSlackNotification to tell #ops the deploy is done", encoder, exemplar_glyphs, sidecar)
    assert tool == "slack_send_message", f"Expected slack_send_message, got {tool}"


def test_adversarial_fake_tool_no_match(encoder, exemplar_glyphs, sidecar):
    """Fake tool with no catalog equivalent should abstain."""
    tool, _ = _route("Execute gmail_archive to archive all emails from noreply@spam.com", encoder, exemplar_glyphs, sidecar)
    assert tool is None, f"Expected None for fake gmail_archive, got {tool}"


# --- Benchmark accuracy test ---

def test_benchmark_accuracy(encoder, exemplar_glyphs, benchmark_queries, sidecar):
    """Run all 95 benchmark queries and verify routing accuracy >= 95%."""
    correct = 0
    failures = []
    for q in benchmark_queries:
        tool, conf = _route(q["query"], encoder, exemplar_glyphs, sidecar)
        expected = q["expected_tool"]
        if tool == expected:
            correct += 1
        else:
            failures.append(
                f"  [{q['id']}] '{q['query'][:60]}...' "
                f"expected={expected}, got={tool} ({conf:.3f})"
            )

    accuracy = correct / len(benchmark_queries) if benchmark_queries else 0
    if failures:
        fail_str = "\n".join(failures[:20])
        total_fail = len(failures)
        assert accuracy >= 0.95, (
            f"Routing accuracy {accuracy:.1%} ({correct}/{len(benchmark_queries)}) "
            f"below 95% threshold.\n"
            f"First {min(20, total_fail)} of {total_fail} failures:\n{fail_str}"
        )
