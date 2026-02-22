#!/usr/bin/env python3
"""
Benchmark: LLM vs Glyphh HDC for SaaS tool routing.

Measures (reported separately for fairness):
  Routing accuracy:
    - Tool accuracy (top-1 match) — comparable across ALL strategies
    - Invalid tool rate (hallucinated tool names)
    - Abstain quality (correctly returning NONE for open-set queries)
  End-to-end validity (only for strategies that produce args):
    - Schema validity (args validated with jsonschema, Draft 7)
    - E2E accuracy (correct tool AND schema-valid args)
  Cost:
    - Latency and token usage

Routing policy: highest-impact (most irreversible action in multi-action queries).
Schema validation: jsonschema Draft 7, enforced even on empty args.
LLM calls: single pass (temperature=0), no majority vote.

Strategies:
  S1 — LLM Only:       query → LLM → tool + args
  S2 — Glyphh Only:    query → Glyphh HDC → tool (no args, schema skipped)
  S3 — Glyphh + LLM:   query → Glyphh HDC → tool, then LLM fills args only
  S4 — LLM + Sidecar:  query → LLM → tool + args, Glyphh confirms/overrides routing

Usage:
    python benchmark/run.py --glyphh-only
    python benchmark/run.py --strategies 1 2 3 4
    python benchmark/run.py --llm-model gpt-4o
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoder import ENCODER_CONFIG, encode_query
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder import Encoder

BENCHMARK_DIR = Path(__file__).parent
QUERIES_PATH = BENCHMARK_DIR / "queries.json"
CATALOG_PATH = BENCHMARK_DIR / "tool_catalog.json"

GLYPHH_THRESHOLD = 0.52


# ═══════════════════════════════════════════════════════════════
# Schema Validation (uses jsonschema for full JSON Schema compliance)
# ═══════════════════════════════════════════════════════════════

def validate_args(tool_name: str, args: dict | None, tools: list[dict]) -> dict:
    """Validate tool args against the catalog schema using jsonschema.

    Always validates — even if args is empty/None, required fields are enforced.
    Returns {"valid": bool, "errors": list[str] | None}.
    """
    import jsonschema

    tool_def = next((t for t in tools if t["name"] == tool_name), None)
    if not tool_def:
        return {"valid": False, "errors": ["tool_not_found"]}

    schema = tool_def.get("schema", {})
    if not schema:
        # No schema defined — nothing to validate against
        return {"valid": True, "errors": None}

    if args is None:
        args = {}

    errors = []

    # Use jsonschema for full validation (types, required, enum, anyOf,
    # additionalProperties, array items, nested objects, etc.)
    validator = jsonschema.Draft7Validator(schema)
    for error in sorted(validator.iter_errors(args), key=lambda e: list(e.path)):
        # Build a readable error string
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        errors.append(f"{path}: {error.message}")

    # Additional check: reject null values for fields whose type doesn't include null
    # (jsonschema handles this for required fields, but we also want to catch
    # optional fields explicitly set to null when the schema doesn't allow it)
    props = schema.get("properties", {})
    for key, val in args.items():
        if val is None and key in props:
            prop_type = props[key].get("type")
            type_allows_null = False
            if isinstance(prop_type, list):
                type_allows_null = "null" in prop_type
            elif prop_type == "null":
                type_allows_null = True
            if not type_allows_null:
                null_err = f"{key}: null value not allowed"
                if null_err not in errors:
                    errors.append(null_err)

    return {"valid": len(errors) == 0, "errors": errors if errors else None}


# ═══════════════════════════════════════════════════════════════
# Glyphh HDC Router
# ═══════════════════════════════════════════════════════════════

class GlyphhRouter:
    def __init__(self, threshold: float = GLYPHH_THRESHOLD):
        self.threshold = threshold
        self.encoder = Encoder(ENCODER_CONFIG)
        self.exemplar_glyphs = []
        self.exemplar_meta: list[dict] = []
        self._load_exemplars()

    def _load_exemplars(self):
        data_path = BENCHMARK_DIR.parent / "data" / "exemplars.jsonl"
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                concept = Concept(
                    name=f"tool_{entry['tool_id']}",
                    attributes={
                        "action": entry["action"],
                        "target": entry["target"],
                        "domain": entry["domain"],
                        "keywords": " ".join(entry["keywords"]) if isinstance(entry["keywords"], list) else entry["keywords"],
                    },
                )
                glyph = self.encoder.encode(concept)
                self.exemplar_glyphs.append(glyph)
                self.exemplar_meta.append(entry)

    def route(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        encoded = encode_query(query)
        q_attrs = encoded["attributes"]

        q_concept = Concept(name="q", attributes=q_attrs)
        q_glyph = self.encoder.encode(q_concept)

        # Role-level weighted similarity
        role_weights = {"action": 1.0, "target": 0.8, "domain": 1.0, "keywords": 0.6}

        q_roles: dict[str, Any] = {}
        for layer in q_glyph.layers.values():
            for seg in layer.segments.values():
                for rname, rvec in seg.roles.items():
                    q_roles[rname] = rvec

        scores: list[tuple[float, int]] = []
        for i, eg in enumerate(self.exemplar_glyphs):
            e_roles: dict[str, Any] = {}
            for layer in eg.layers.values():
                for seg in layer.segments.values():
                    for rname, rvec in seg.roles.items():
                        e_roles[rname] = rvec

            weighted_sum = 0.0
            weight_total = 0.0
            for rname, w in role_weights.items():
                if rname in q_roles and rname in e_roles:
                    sim = float(cosine_similarity(q_roles[rname].data, e_roles[rname].data))
                    weighted_sum += sim * w
                    weight_total += w

            score = weighted_sum / weight_total if weight_total > 0 else 0.0
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        top_3 = [{"tool": self.exemplar_meta[i]["tool_id"], "score": round(s, 4)} for s, i in scores[:3]]

        if scores and scores[0][0] >= self.threshold:
            best_score, best_idx = scores[0]
            return {
                "tool": self.exemplar_meta[best_idx]["tool_id"],
                "confidence": round(best_score, 4),
                "latency_ms": elapsed_ms,
                "top_3": top_3,
                "args": {},
            }
        return {
            "tool": None,
            "confidence": round(scores[0][0], 4) if scores else 0.0,
            "latency_ms": elapsed_ms,
            "top_3": top_3,
            "args": {},
        }


# ═══════════════════════════════════════════════════════════════
# LLM Helpers
# ═══════════════════════════════════════════════════════════════

_openai_client = None

def _get_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _llm_call(messages: list[dict], model: str, max_tokens: int = 500) -> dict:
    client = _get_client()
    start = time.perf_counter()
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0, messages=messages, max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        parsed = json.loads(content)
        usage = resp.usage
        return {
            "parsed": parsed, "latency_ms": elapsed_ms,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "raw": content, "error": None,
        }
    except json.JSONDecodeError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"parsed": {}, "latency_ms": elapsed_ms, "prompt_tokens": 0, "completion_tokens": 0, "raw": content, "error": "json_parse_error"}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"parsed": {}, "latency_ms": elapsed_ms, "prompt_tokens": 0, "completion_tokens": 0, "raw": "", "error": str(e)}


def _build_tool_list(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        schema = t.get("schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        any_of = schema.get("anyOf", [])
        params = []
        for pname, pdef in props.items():
            req = " REQUIRED" if pname in required else ""
            ptype = pdef.get("type", "any")
            desc = pdef.get("description", "")
            enum = pdef.get("enum")
            extra = f" enum={enum}" if enum else ""
            params.append(f"    {pname} ({ptype}{req}){extra}: {desc}")
        if any_of:
            constraints = " OR ".join(
                "(" + ", ".join(c.get("required", [])) + ")" for c in any_of
            )
            params.append(f"    [anyOf constraint: at least one of {constraints} must be provided]")
        param_block = "\n".join(params) if params else "    (no parameters)"
        lines.append(f"- {t['name']}: {t['description']}\n  Parameters:\n{param_block}")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Strategy Prompts
# ═══════════════════════════════════════════════════════════════

S1_SYSTEM = (
    "You are a tool router for a SaaS automation platform. Given a user query, "
    "select the SINGLE most appropriate tool from the catalog below and provide "
    "the arguments to call it with.\n\n"
    "TOOL CATALOG:\n{tools}\n\n"
    "ROUTING POLICY:\n"
    "For multi-action requests, select the tool for the highest-impact or most "
    "irreversible action. Secondary actions (notifications, logging) are deferred.\n\n"
    "RULES:\n"
    "1. You MUST select a tool from the catalog above or return null. "
    "NEVER invent tool names that are not in the catalog.\n"
    "2. If the user mentions a tool name that does NOT exist in the catalog, "
    "find the closest REAL tool that matches the intent, or return null if none fits.\n"
    "3. Return null for queries that don't map to any tool in the catalog "
    "(greetings, general questions, requests for tools that don't exist).\n"
    "4. Arguments must match the tool's schema exactly:\n"
    "   - Use correct types (string, integer, array, etc.)\n"
    "   - Dollar amounts for Stripe must be in cents (e.g. $49.99 = 4999)\n"
    "   - Use enum values exactly as specified\n"
    "   - Include all required fields\n"
    "   - Do not add fields not in the schema\n"
    "   - Do NOT include optional fields with null values — omit them entirely if unknown\n"
    "   - If a tool has an anyOf constraint, at least one of the listed field groups must be provided\n"
    "5. If you cannot determine a required argument from the query, "
    "use a reasonable placeholder but still select the correct tool.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"tool": "<tool_name or null>", "confidence": <0.0-1.0>, '
    '"args": {{<arguments matching the tool schema>}}, '
    '"reasoning": "<brief explanation>"}}'
)

S3_SYSTEM = (
    "You are an argument extractor for a SaaS automation platform. "
    "The tool has already been selected for you: {tool_name}\n\n"
    "TOOL DEFINITION:\n{tool_def}\n\n"
    "Given the user's query, generate ONLY the arguments for this tool. "
    "Do NOT select a different tool.\n\n"
    "RULES:\n"
    "1. Arguments must match the tool's schema exactly.\n"
    "2. Use correct types (string, integer, array, etc.)\n"
    "3. Dollar amounts for Stripe must be in cents (e.g. $49.99 = 4999)\n"
    "4. Use enum values exactly as specified\n"
    "5. Include all required fields\n"
    "6. Do not add fields not in the schema\n"
    "7. Do NOT include optional fields with null values — omit them entirely if unknown\n"
    "8. If a tool has an anyOf constraint, at least one of the listed field groups must be provided\n"
    "9. If you cannot determine a required argument, use a reasonable placeholder.\n\n"
    "Respond with ONLY valid JSON (the args object):\n"
    "{{<arguments matching the tool schema>}}"
)


# ═══════════════════════════════════════════════════════════════
# Strategy Implementations
# ═══════════════════════════════════════════════════════════════

def strategy_1(query: str, tools: list[dict], model: str) -> dict:
    """S1: LLM picks tool + generates args."""
    system = S1_SYSTEM.format(tools=_build_tool_list(tools))
    r = _llm_call([{"role": "system", "content": system}, {"role": "user", "content": query}], model)
    p = r["parsed"]
    tool = p.get("tool")
    if tool in ("null", "None", "none", ""):
        tool = None
    args = p.get("args", {})
    if args is None:
        args = {}
    return {
        "tool": tool,
        "confidence": p.get("confidence", 0.0),
        "args": args,
        "reasoning": p.get("reasoning", ""),
        "latency_ms": r["latency_ms"],
        "tokens": r["prompt_tokens"] + r["completion_tokens"],
        "error": r["error"],
    }


def strategy_2(query: str, router: GlyphhRouter) -> dict:
    """S2: Glyphh picks tool, no args."""
    r = router.route(query)
    return {
        "tool": r["tool"],
        "confidence": r["confidence"],
        "args": {},
        "latency_ms": r["latency_ms"],
        "tokens": 0,
        "top_3": r["top_3"],
    }


def _build_single_tool_def(tool_name: str, tools: list[dict]) -> str:
    """Build a formatted definition for a single tool (used by S3)."""
    tool_def = next((t for t in tools if t["name"] == tool_name), None)
    if not tool_def:
        return f"Tool '{tool_name}' not found."
    schema = tool_def.get("schema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])
    any_of = schema.get("anyOf", [])
    lines = [f"{tool_name}: {tool_def['description']}", "Parameters:"]
    for pname, pdef in props.items():
        req = " REQUIRED" if pname in required else ""
        ptype = pdef.get("type", "any")
        desc = pdef.get("description", "")
        enum = pdef.get("enum")
        extra = f" enum={enum}" if enum else ""
        lines.append(f"  {pname} ({ptype}{req}){extra}: {desc}")
    if any_of:
        constraints = " OR ".join(
            "(" + ", ".join(c.get("required", [])) + ")" for c in any_of
        )
        lines.append(f"  [anyOf constraint: at least one of {constraints} must be provided]")
    return "\n".join(lines)


def strategy_3(query: str, router: GlyphhRouter, tools: list[dict], model: str) -> dict:
    """S3: Glyphh picks tool, LLM fills args only."""
    # Step 1: Glyphh routes
    route_result = router.route(query)
    glyphh_tool = route_result["tool"]
    glyphh_latency = route_result["latency_ms"]

    if glyphh_tool is None:
        # Glyphh abstained — no LLM call needed
        return {
            "tool": None,
            "confidence": route_result["confidence"],
            "args": {},
            "latency_ms": glyphh_latency,
            "tokens": 0,
            "top_3": route_result["top_3"],
            "routing_source": "glyphh",
        }

    # Step 2: LLM generates args for the fixed tool
    tool_def_str = _build_single_tool_def(glyphh_tool, tools)
    system = S3_SYSTEM.format(tool_name=glyphh_tool, tool_def=tool_def_str)
    r = _llm_call([{"role": "system", "content": system}, {"role": "user", "content": query}], model)

    args = r["parsed"] if isinstance(r["parsed"], dict) else {}
    # If the LLM wrapped it in {"args": {...}}, unwrap
    if "args" in args and isinstance(args["args"], dict) and len(args) <= 3:
        args = args["args"]

    return {
        "tool": glyphh_tool,
        "confidence": route_result["confidence"],
        "args": args,
        "latency_ms": glyphh_latency + r["latency_ms"],
        "tokens": r["prompt_tokens"] + r["completion_tokens"],
        "top_3": route_result["top_3"],
        "routing_source": "glyphh",
        "llm_latency_ms": r["latency_ms"],
        "error": r["error"],
    }


def strategy_4(query: str, router: GlyphhRouter, tools: list[dict], model: str) -> dict:
    """S4: LLM picks tool + args, Glyphh sidecar confirms or overrides routing.

    Flow:
      1. LLM selects tool + generates args (same as S1)
      2. Glyphh independently routes the query
      3. If Glyphh agrees with LLM -> keep LLM result as-is
         If Glyphh picks a different tool -> override to Glyphh's tool, re-run LLM for args
         If Glyphh abstains (None) and LLM picked a tool -> keep LLM (Glyphh may be conservative)
         If both abstain -> return None
    """
    # Step 1: LLM does full routing + args
    llm_result = strategy_1(query, tools, model)
    llm_tool = llm_result["tool"]
    llm_latency = llm_result["latency_ms"]
    llm_tokens = llm_result["tokens"]

    # Step 2: Glyphh routes independently
    route_result = router.route(query)
    glyphh_tool = route_result["tool"]
    glyphh_latency = route_result["latency_ms"]

    total_latency = llm_latency + glyphh_latency
    total_tokens = llm_tokens

    # Step 3: Reconcile
    if llm_tool is None and glyphh_tool is None:
        # Both abstain
        return {
            "tool": None,
            "confidence": route_result["confidence"],
            "args": {},
            "latency_ms": total_latency,
            "tokens": total_tokens,
            "top_3": route_result["top_3"],
            "routing_source": "both_abstain",
            "llm_tool": llm_tool,
            "glyphh_tool": glyphh_tool,
            "sidecar_action": "both_abstain",
            "error": llm_result.get("error"),
        }

    if glyphh_tool is None:
        # Glyphh abstains but LLM picked something — trust LLM
        return {
            **llm_result,
            "latency_ms": total_latency,
            "routing_source": "llm_only",
            "llm_tool": llm_tool,
            "glyphh_tool": None,
            "sidecar_action": "glyphh_abstain_keep_llm",
        }

    if llm_tool is None:
        # LLM abstains but Glyphh picked something — use Glyphh + get args
        tool_def_str = _build_single_tool_def(glyphh_tool, tools)
        system = S3_SYSTEM.format(tool_name=glyphh_tool, tool_def=tool_def_str)
        r = _llm_call([{"role": "system", "content": system}, {"role": "user", "content": query}], model)
        args = r["parsed"] if isinstance(r["parsed"], dict) else {}
        if "args" in args and isinstance(args["args"], dict) and len(args) <= 3:
            args = args["args"]
        return {
            "tool": glyphh_tool,
            "confidence": route_result["confidence"],
            "args": args,
            "latency_ms": total_latency + r["latency_ms"],
            "tokens": total_tokens + r["prompt_tokens"] + r["completion_tokens"],
            "top_3": route_result["top_3"],
            "routing_source": "glyphh_override",
            "llm_tool": None,
            "glyphh_tool": glyphh_tool,
            "sidecar_action": "llm_abstain_use_glyphh",
            "error": r["error"],
        }

    if llm_tool == glyphh_tool:
        # Agreement — keep LLM result (has args already)
        return {
            **llm_result,
            "latency_ms": total_latency,
            "routing_source": "agreement",
            "llm_tool": llm_tool,
            "glyphh_tool": glyphh_tool,
            "sidecar_action": "agree",
        }

    # Disagreement — Glyphh overrides, re-run LLM for args on corrected tool
    tool_def_str = _build_single_tool_def(glyphh_tool, tools)
    system = S3_SYSTEM.format(tool_name=glyphh_tool, tool_def=tool_def_str)
    r = _llm_call([{"role": "system", "content": system}, {"role": "user", "content": query}], model)
    args = r["parsed"] if isinstance(r["parsed"], dict) else {}
    if "args" in args and isinstance(args["args"], dict) and len(args) <= 3:
        args = args["args"]
    return {
        "tool": glyphh_tool,
        "confidence": route_result["confidence"],
        "args": args,
        "latency_ms": total_latency + r["latency_ms"],
        "tokens": total_tokens + r["prompt_tokens"] + r["completion_tokens"],
        "top_3": route_result["top_3"],
        "routing_source": "glyphh_override",
        "llm_tool": llm_tool,
        "glyphh_tool": glyphh_tool,
        "sidecar_action": "override",
        "error": r["error"],
    }


# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════

def score_result(
    result_tool: str | None,
    result_args: dict,
    expected_tool: str | None,
    expected_args: dict | None,
    valid_tool_names: set[str],
    all_tools: list[dict],
    skip_schema: bool = False,
) -> dict:
    """Score a single result across all dimensions."""
    scores = {
        "tool_correct": False,
        "tool_label": "wrong",
        "invalid_tool": False,
        "schema_valid": True,
        "schema_errors": None,
        "abstain_correct": None,
    }

    # Tool accuracy
    if expected_tool is None:
        scores["abstain_correct"] = result_tool is None
        if result_tool is None:
            scores["tool_label"] = "correct_abstain"
            scores["tool_correct"] = True
        elif result_tool not in valid_tool_names:
            scores["tool_label"] = "hallucinated_tool"
            scores["invalid_tool"] = True
        else:
            scores["tool_label"] = "false_positive"
    else:
        if result_tool is None:
            scores["tool_label"] = "false_abstain"
        elif result_tool == expected_tool:
            scores["tool_label"] = "correct"
            scores["tool_correct"] = True
        elif result_tool not in valid_tool_names:
            scores["tool_label"] = "hallucinated_tool"
            scores["invalid_tool"] = True
        else:
            scores["tool_label"] = "wrong_tool"

    # Schema validation (only if a tool was selected and not skip_schema)
    if not skip_schema and result_tool and result_tool in valid_tool_names:
        validation = validate_args(result_tool, result_args, all_tools)
        scores["schema_valid"] = validation["valid"]
        scores["schema_errors"] = validation.get("errors")

    return scores


# ═══════════════════════════════════════════════════════════════
# Aggregation & Reporting
# ═══════════════════════════════════════════════════════════════

def _aggregate(results: list[dict], name: str, skip_schema: bool = False) -> dict:
    total = len(results)
    if total == 0:
        return {"name": name, "total": 0}

    tool_correct = sum(1 for r in results if r["tool_correct"])
    invalid_tools = sum(1 for r in results if r["invalid_tool"])

    # Schema: count all results where a known tool was selected (even if args empty)
    schema_checked = [r for r in results if r.get("result_tool") and not skip_schema]
    schema_valid = sum(1 for r in schema_checked if r["schema_valid"])

    # End-to-end: tool correct AND schema valid (only for strategies that produce args)
    e2e_candidates = [r for r in results if r.get("expected_tool") is not None and not skip_schema]
    e2e_correct = sum(1 for r in e2e_candidates if r["tool_correct"] and r["schema_valid"])

    abstain_queries = [r for r in results if r["expected_tool"] is None]
    abstain_correct = sum(1 for r in abstain_queries if r.get("abstain_correct", False))

    in_scope = [r for r in results if r["expected_tool"] is not None]
    in_scope_correct = sum(1 for r in in_scope if r["tool_correct"])

    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r.get("category", "unknown"), []).append(r)

    cat_breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        cat_correct = sum(1 for r in cat_results if r["tool_correct"])
        cat_invalid = sum(1 for r in cat_results if r["invalid_tool"])
        cat_schema = [r for r in cat_results if r.get("result_tool") and not skip_schema]
        cat_schema_valid = sum(1 for r in cat_schema if r["schema_valid"])
        cat_e2e = [r for r in cat_results if r.get("expected_tool") is not None and not skip_schema]
        cat_e2e_correct = sum(1 for r in cat_e2e if r["tool_correct"] and r["schema_valid"])
        cat_breakdown[cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": cat_correct / len(cat_results) if cat_results else 0,
            "invalid_tools": cat_invalid,
            "schema_valid": cat_schema_valid,
            "schema_total": len(cat_schema),
            "e2e_correct": cat_e2e_correct,
            "e2e_total": len(cat_e2e),
        }

    latencies = [r["latency_ms"] for r in results]
    tokens = [r.get("tokens", 0) for r in results]

    return {
        "name": name,
        "total": total,
        "tool_accuracy": tool_correct / total if total else 0,
        "tool_correct": tool_correct,
        "invalid_tool_rate": invalid_tools / total if total else 0,
        "invalid_tools": invalid_tools,
        "schema_validity": schema_valid / len(schema_checked) if schema_checked else None,
        "schema_valid": schema_valid,
        "schema_total": len(schema_checked),
        "skip_schema": skip_schema,
        "e2e_accuracy": e2e_correct / len(e2e_candidates) if e2e_candidates else None,
        "e2e_correct": e2e_correct,
        "e2e_total": len(e2e_candidates),
        "abstain_accuracy": abstain_correct / len(abstain_queries) if abstain_queries else 1.0,
        "abstain_correct": abstain_correct,
        "abstain_total": len(abstain_queries),
        "in_scope_accuracy": in_scope_correct / len(in_scope) if in_scope else 0,
        "in_scope_correct": in_scope_correct,
        "in_scope_total": len(in_scope),
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "tokens_total": sum(tokens),
        "tokens_mean": sum(tokens) / len(tokens) if tokens else 0,
        "categories": cat_breakdown,
    }


def _print_report(strategies: list[dict]):
    print("\n" + "=" * 100)
    print("  SAAS TOOL ROUTER BENCHMARK — RESULTS")
    print("=" * 100)

    # Section 1: Routing accuracy (apples-to-apples across all strategies)
    print("\n  ROUTING ACCURACY (tool selection only — comparable across all strategies)")
    header = f"  {'Strategy':<30} {'ToolAcc':>8} {'InScope':>8} {'Abstain':>8} {'Invalid':>8} {'Lat(ms)':>8} {'Tokens':>7}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for s in strategies:
        print(
            f"  {s['name']:<30} "
            f"{s['tool_accuracy']:>7.1%} "
            f"{s['in_scope_accuracy']:>7.1%} "
            f"{s['abstain_accuracy']:>7.1%} "
            f"{s['invalid_tool_rate']:>7.1%} "
            f"{s['latency_mean_ms']:>7.1f} "
            f"{s['tokens_total']:>7}"
        )

    # Section 2: End-to-end (tool + schema valid args) — only for strategies that produce args
    e2e_strats = [s for s in strategies if not s.get("skip_schema")]
    if e2e_strats:
        print("\n  END-TO-END VALIDITY (correct tool + schema-valid args)")
        header2 = f"  {'Strategy':<30} {'E2E Acc':>8} {'Schema':>8} {'SchemaOK':>12}"
        print(f"\n{header2}")
        print("  " + "-" * (len(header2) - 2))
        for s in e2e_strats:
            schema_str = f"{s['schema_validity']:>7.1%}" if s.get("schema_validity") is not None else "  N/A  "
            e2e_str = f"{s['e2e_accuracy']:>7.1%}" if s.get("e2e_accuracy") is not None else "  N/A  "
            schema_detail = f"{s['schema_valid']}/{s['schema_total']}"
            print(
                f"  {s['name']:<30} "
                f"{e2e_str} "
                f"{schema_str} "
                f"{schema_detail:>12}"
            )

    # Per-category routing breakdown
    print(f"\n  ROUTING BY CATEGORY")
    print(f"  {'Category':<23}", end="")
    for s in strategies:
        print(f" {s['name'][:18]:>18}", end="")
    print()
    print("  " + "-" * (23 + 19 * len(strategies)))

    all_cats = sorted({cat for s in strategies for cat in s.get("categories", {})})
    for cat in all_cats:
        print(f"    {cat:<21}", end="")
        for s in strategies:
            cd = s.get("categories", {}).get(cat, {})
            acc = cd.get("accuracy", 0)
            n = cd.get("total", 0)
            inv = cd.get("invalid_tools", 0)
            inv_str = f" h={inv}" if inv > 0 else ""
            print(f" {acc:>6.1%} ({n:>2}){inv_str:>5}", end="")
        print()

    # Schema failure detail
    schema_strats = [s for s in strategies if not s.get("skip_schema") and s.get("schema_validity") is not None]
    if schema_strats:
        has_failures = any(s['schema_valid'] < s['schema_total'] for s in schema_strats)
        if has_failures:
            print(f"\n  SCHEMA FAILURES:")
            for s in schema_strats:
                if s['schema_valid'] < s['schema_total']:
                    print(f"    {s['name']}: {s['schema_valid']}/{s['schema_total']} passed ({s['schema_validity']:.1%})")

    print("\n" + "=" * 100)


def _progress(current: int, total: int, label: str = ""):
    pct = current / total if total else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {label} [{bar}] {current}/{total}", end="", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════

def run_benchmark(
    strategies: list[int] | None = None,
    glyphh_only: bool = False,
    llm_model: str = "gpt-4o-mini",
    output_dir: str | None = None,
):
    with open(QUERIES_PATH) as f:
        query_data = json.load(f)
    queries = query_data["queries"]

    with open(CATALOG_PATH) as f:
        catalog_data = json.load(f)
    tools = catalog_data["tools"]
    valid_tool_names = {t["name"] for t in tools}

    if glyphh_only:
        active = [2]
    elif strategies:
        active = strategies
    else:
        active = [1, 2]

    llm_strategies = {1, 3, 4}
    if llm_strategies & set(active) and not os.environ.get("OPENAI_API_KEY"):
        print("⚠  OPENAI_API_KEY not set — skipping LLM strategies")
        active = [s for s in active if s not in llm_strategies]
        if not active:
            print("No strategies to run.")
            return

    router = GlyphhRouter()
    print(f"Loaded {len(router.exemplar_meta)} exemplars, {len(queries)} queries, {len(tools)} tools")
    print(f"Strategies: {active}  |  LLM: {llm_model}\n")

    strategy_names = {
        1: "S1: LLM Only",
        2: "S2: Glyphh Only",
        3: "S3: Glyphh Route + LLM Args",
        4: "S4: LLM + Glyphh Sidecar",
    }
    # S2 is routing-only — schema validation doesn't apply
    skip_schema_for = {2}

    all_strategy_results = []

    for strat_num in active:
        name = strategy_names[strat_num]
        print(f"\n── {name} ──")
        results = []

        for qi, q in enumerate(queries):
            _progress(qi + 1, len(queries), name[:25])

            if strat_num == 1:
                r = strategy_1(q["query"], tools, llm_model)
            elif strat_num == 2:
                r = strategy_2(q["query"], router)
            elif strat_num == 3:
                r = strategy_3(q["query"], router, tools, llm_model)
            elif strat_num == 4:
                r = strategy_4(q["query"], router, tools, llm_model)
            else:
                continue

            skip_schema = strat_num in skip_schema_for
            scoring = score_result(
                r["tool"], r.get("args", {}),
                q["expected_tool"], q.get("expected_args"),
                valid_tool_names, tools,
                skip_schema=skip_schema,
            )

            results.append({
                "query_id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "expected_tool": q["expected_tool"],
                "result_tool": r["tool"],
                "result_args": r.get("args", {}),
                "confidence": r.get("confidence", 0),
                "latency_ms": r.get("latency_ms", 0),
                "tokens": r.get("tokens", 0),
                **scoring,
                **{k: v for k, v in r.items() if k not in ("tool", "confidence", "latency_ms", "tokens", "args")},
            })

        print()  # newline after progress bar
        agg = _aggregate(results, name, skip_schema=strat_num in skip_schema_for)
        agg["raw_results"] = results
        all_strategy_results.append(agg)

    _print_report(all_strategy_results)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for agg in all_strategy_results:
            safe_name = agg["name"].replace(" ", "_").replace(":", "").replace("+", "plus")
            with open(out_path / f"{safe_name}.json", "w") as f:
                json.dump(agg, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}/")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SaaS Tool Router Benchmark: Glyphh HDC vs LLM")
    parser.add_argument("--strategies", nargs="+", type=int, choices=[1, 2, 3, 4], help="Which strategies to run")
    parser.add_argument("--glyphh-only", action="store_true", help="Run only Strategy 2 (no API key needed)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--output", type=str, help="Directory to save raw JSON results")
    args = parser.parse_args()

    run_benchmark(
        strategies=args.strategies,
        glyphh_only=args.glyphh_only,
        llm_model=args.llm_model,
        output_dir=args.output,
    )
