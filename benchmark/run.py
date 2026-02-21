#!/usr/bin/env python3
"""
Benchmark: Four strategies for tool routing.

Strategy 1 — LLM Only:        query → LLM → result
Strategy 2 — Glyphh Only:     query → Glyphh HDC → result
Strategy 3 — LLM→Glyphh→LLM:  query → LLM extracts intent → Glyphh routes → LLM confirms
Strategy 4 — LLM→Glyphh→ASK→LLM→Glyphh→LLM:  same as 3 + clarification loop on low confidence

Usage:
    python benchmark/run.py
    python benchmark/run.py --glyphh-only
    python benchmark/run.py --strategies 1 2 3
    python benchmark/run.py --llm-model gpt-4o
    python benchmark/run.py --output benchmark/results/
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

GLYPHH_THRESHOLD = 0.59
GLYPHH_LOW_CONFIDENCE = 0.55
LLM_RUNS_PER_QUERY = 3


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
                        "verb": entry["verb"],
                        "object": entry["object"],
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
        oos_qualifiers = encoded.get("out_of_scope_qualifiers", set())
        result = self._route_attrs(q_attrs, start)
        # If the query contains out-of-scope qualifiers, force rejection
        if oos_qualifiers and result["tool_id"] is not None:
            result["tool_id"] = None
            result["rejected_by"] = "oos_qualifier"
            result["oos_qualifiers"] = list(oos_qualifiers)
        return result

    def route_from_intent(self, intent: dict[str, str]) -> dict[str, Any]:
        start = time.perf_counter()
        attrs = {
            "verb": intent.get("verb", "query"),
            "object": intent.get("object", "general"),
            "domain": intent.get("domain", "release"),
            "keywords": intent.get("keywords", ""),
        }
        return self._route_attrs(attrs, start)

    def _route_attrs(self, attrs: dict, start: float) -> dict[str, Any]:
        q_concept = Concept(name="q", attributes=attrs)
        q_glyph = self.encoder.encode(q_concept)

        # Use role-level similarity with configured weights.
        # This respects the encoder config: verb=1.0, object=0.8, domain=1.0
        # and differentiates tools that share verb+object but differ on domain.
        role_weights = {
            "verb": 1.0,
            "object": 0.8,
            "domain": 1.0,
            "keywords": 0.6,
        }

        # Collect query role vectors from all segments
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
        top_3 = [{"tool_id": self.exemplar_meta[i]["tool_id"], "score": s} for s, i in scores[:3]]
        if scores and scores[0][0] >= self.threshold:
            best_score, best_idx = scores[0]
            return {"tool_id": self.exemplar_meta[best_idx]["tool_id"], "confidence": best_score, "latency_ms": elapsed_ms, "top_3": top_3}
        return {"tool_id": None, "confidence": scores[0][0] if scores else 0.0, "latency_ms": elapsed_ms, "top_3": top_3}


# ═══════════════════════════════════════════════════════════════
# LLM Helpers
# ═══════════════════════════════════════════════════════════════

def _get_client():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _llm_call(messages: list[dict], model: str, max_tokens: int = 200) -> dict:
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
        return {"parsed": parsed, "latency_ms": elapsed_ms, "prompt_tokens": usage.prompt_tokens if usage else 0, "completion_tokens": usage.completion_tokens if usage else 0, "raw": content, "error": None}
    except json.JSONDecodeError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"parsed": {}, "latency_ms": elapsed_ms, "prompt_tokens": 0, "completion_tokens": 0, "raw": content, "error": "json_parse_error"}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"parsed": {}, "latency_ms": elapsed_ms, "prompt_tokens": 0, "completion_tokens": 0, "raw": "", "error": str(e)}


def _build_tool_list(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        params = ""
        if t.get("parameters"):
            ps = [f"{p['name']} ({p['type']})" for p in t["parameters"]]
            params = f" | params: {', '.join(ps)}"
        lines.append(f"- {t['id']}: {t['description']}{params}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Strategy Prompts
# ═══════════════════════════════════════════════════════════════

S1_SYSTEM = (
    "You are a tool router. Given a user query, select the most appropriate tool "
    "from the catalog below. If no tool matches, respond with tool_id: null.\n\n"
    "TOOL CATALOG:\n{tools}\n\n"
    "IMPORTANT: Only select tools that exist in the catalog above. "
    "If the query doesn't match any tool, you MUST return null.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"tool_id": "<tool_id or null>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}'
)

S3_EXTRACT = (
    "Extract the structured intent from this user query for tool routing.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"verb": "<build|deploy|query|manage>", "object": "<target>", '
    '"domain": "<release|test|build|docker>", "keywords": "<space-separated>"}}'
)

S3_CONFIRM = (
    "A tool router selected the following tool based on the user's query.\n\n"
    "User query: {query}\n"
    "Selected tool: {tool_id} (confidence: {confidence:.2f})\n"
    "Top candidates: {top_3}\n\n"
    "TOOL CATALOG:\n{tools}\n\n"
    "Confirm or correct the selection. If the tool is wrong or the query is out of scope, "
    "return null. Respond with ONLY valid JSON:\n"
    '{{"tool_id": "<tool_id or null>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}'
)

S4_ASK = (
    "The tool router is uncertain about this query.\n\n"
    "User query: {query}\n"
    "Best match: {tool_id} (confidence: {confidence:.2f})\n"
    "Top candidates: {top_3}\n\n"
    "Generate ONE short clarifying question to disambiguate the user's intent.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"question": "<clarifying question>"}}'
)

S4_ANSWER = (
    "A user asked: {original_query}\n"
    "The system asked for clarification: {clarifying_question}\n\n"
    "Simulate a reasonable user response that clarifies their intent. Be concise.\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"answer": "<simulated user response>"}}'
)

S4_REEXTRACT = (
    "Extract the structured intent from this conversation for tool routing.\n\n"
    "Original query: {original_query}\n"
    "Clarification Q: {question}\n"
    "User answer: {answer}\n\n"
    "Respond with ONLY valid JSON:\n"
    '{{"verb": "<build|deploy|query|manage>", "object": "<target>", '
    '"domain": "<release|test|build|docker>", "keywords": "<space-separated>"}}'
)


# ═══════════════════════════════════════════════════════════════
# Strategy Implementations
# ═══════════════════════════════════════════════════════════════

def strategy_1(query: str, tools: list[dict], model: str) -> dict:
    system = S1_SYSTEM.format(tools=_build_tool_list(tools))
    r = _llm_call([{"role": "system", "content": system}, {"role": "user", "content": query}], model)
    p = r["parsed"]
    return {"tool_id": p.get("tool_id"), "confidence": p.get("confidence", 0.0), "reasoning": p.get("reasoning", ""), "latency_ms": r["latency_ms"], "tokens": r["prompt_tokens"] + r["completion_tokens"], "error": r["error"]}


def strategy_2(query: str, router: GlyphhRouter) -> dict:
    r = router.route(query)
    return {"tool_id": r["tool_id"], "confidence": r["confidence"], "latency_ms": r["latency_ms"], "tokens": 0, "top_3": r["top_3"]}


def strategy_3(query: str, tools: list[dict], router: GlyphhRouter, model: str) -> dict:
    lat, tok = 0.0, 0
    # Extract intent
    ext = _llm_call([{"role": "system", "content": S3_EXTRACT}, {"role": "user", "content": query}], model)
    lat += ext["latency_ms"]; tok += ext["prompt_tokens"] + ext["completion_tokens"]
    intent = ext["parsed"]
    # Glyphh route
    gr = router.route_from_intent(intent) if intent and not ext["error"] else router.route(query)
    lat += gr["latency_ms"]
    # Confirm
    top_3_str = ", ".join(f"{t['tool_id']}({t['score']:.2f})" for t in gr["top_3"])
    conf = _llm_call([{"role": "system", "content": S3_CONFIRM.format(query=query, tool_id=gr["tool_id"] or "null", confidence=gr["confidence"], top_3=top_3_str, tools=_build_tool_list(tools))}, {"role": "user", "content": "confirm"}], model)
    lat += conf["latency_ms"]; tok += conf["prompt_tokens"] + conf["completion_tokens"]
    p = conf["parsed"]
    return {"tool_id": p.get("tool_id"), "confidence": p.get("confidence", 0.0), "reasoning": p.get("reasoning", ""), "latency_ms": lat, "tokens": tok, "glyphh_tool": gr["tool_id"], "glyphh_confidence": gr["confidence"]}


def strategy_4(query: str, tools: list[dict], router: GlyphhRouter, model: str) -> dict:
    lat, tok, steps = 0.0, 0, []
    # Extract intent
    ext = _llm_call([{"role": "system", "content": S3_EXTRACT}, {"role": "user", "content": query}], model)
    lat += ext["latency_ms"]; tok += ext["prompt_tokens"] + ext["completion_tokens"]; steps.append("extract")
    intent = ext["parsed"]
    # Glyphh route 1
    gr = router.route_from_intent(intent) if intent and not ext["error"] else router.route(query)
    lat += gr["latency_ms"]; steps.append("glyphh_1")
    asked = False
    # ASK if low confidence
    if gr["confidence"] < GLYPHH_LOW_CONFIDENCE:
        asked = True
        top_3_str = ", ".join(f"{t['tool_id']}({t['score']:.2f})" for t in gr["top_3"])
        ask = _llm_call([{"role": "system", "content": S4_ASK.format(query=query, tool_id=gr["tool_id"] or "null", confidence=gr["confidence"], top_3=top_3_str)}, {"role": "user", "content": "ask"}], model)
        lat += ask["latency_ms"]; tok += ask["prompt_tokens"] + ask["completion_tokens"]; steps.append("ask")
        question = ask["parsed"].get("question", "Could you be more specific?")
        # Simulate answer
        ans = _llm_call([{"role": "system", "content": S4_ANSWER.format(original_query=query, clarifying_question=question)}, {"role": "user", "content": "answer"}], model)
        lat += ans["latency_ms"]; tok += ans["prompt_tokens"] + ans["completion_tokens"]; steps.append("answer")
        answer = ans["parsed"].get("answer", query)
        # Re-extract
        reext = _llm_call([{"role": "system", "content": S4_REEXTRACT.format(original_query=query, question=question, answer=answer)}, {"role": "user", "content": "extract"}], model)
        lat += reext["latency_ms"]; tok += reext["prompt_tokens"] + reext["completion_tokens"]; steps.append("re_extract")
        new_intent = reext["parsed"]
        # Glyphh route 2
        gr = router.route_from_intent(new_intent) if new_intent and not reext["error"] else router.route(query)
        lat += gr["latency_ms"]; steps.append("glyphh_2")
    # Confirm
    top_3_str = ", ".join(f"{t['tool_id']}({t['score']:.2f})" for t in gr["top_3"])
    conf = _llm_call([{"role": "system", "content": S3_CONFIRM.format(query=query, tool_id=gr["tool_id"] or "null", confidence=gr["confidence"], top_3=top_3_str, tools=_build_tool_list(tools))}, {"role": "user", "content": "confirm"}], model)
    lat += conf["latency_ms"]; tok += conf["prompt_tokens"] + conf["completion_tokens"]; steps.append("confirm")
    p = conf["parsed"]
    return {"tool_id": p.get("tool_id"), "confidence": p.get("confidence", 0.0), "reasoning": p.get("reasoning", ""), "latency_ms": lat, "tokens": tok, "glyphh_tool": gr["tool_id"], "glyphh_confidence": gr["confidence"], "asked": asked, "steps": steps}


# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════

def score_result(result_tool_id: str | None, expected_tool_id: str | None, valid_tool_ids: set[str]) -> str:
    """
    Returns one of:
      'correct'       — matched expected tool
      'refusal_ok'    — correctly returned None for out-of-scope
      'refusal_bad'   — returned None when a tool was expected
      'hallucination' — returned a tool_id not in the catalog
      'wrong'         — returned a valid but incorrect tool
    """
    if expected_tool_id is None:
        return "refusal_ok" if result_tool_id is None else "wrong"
    if result_tool_id is None:
        return "refusal_bad"
    if result_tool_id == expected_tool_id:
        return "correct"
    if result_tool_id not in valid_tool_ids:
        return "hallucination"
    return "wrong"


# ═══════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════

def _aggregate(results: list[dict], name: str) -> dict:
    total = len(results)
    if total == 0:
        return {"name": name, "total": 0}

    correct = sum(1 for r in results if r["score_label"] == "correct")
    refusal_ok = sum(1 for r in results if r["score_label"] == "refusal_ok")
    refusal_bad = sum(1 for r in results if r["score_label"] == "refusal_bad")
    hallucinations = sum(1 for r in results if r["score_label"] == "hallucination")
    wrong = sum(1 for r in results if r["score_label"] == "wrong")

    # Queries that SHOULD match a tool
    in_scope = [r for r in results if r["expected_tool"] is not None]
    # Queries that should NOT match
    out_scope = [r for r in results if r["expected_tool"] is None]

    accuracy = (correct + refusal_ok) / total if total else 0
    in_scope_acc = correct / len(in_scope) if in_scope else 0
    refusal_acc = refusal_ok / len(out_scope) if out_scope else 0
    hallucination_rate = hallucinations / total if total else 0

    latencies = [r["latency_ms"] for r in results]
    tokens = [r.get("tokens", 0) for r in results]

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    cat_breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        cat_correct = sum(1 for r in cat_results if r["score_label"] in ("correct", "refusal_ok"))
        cat_breakdown[cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": cat_correct / len(cat_results) if cat_results else 0,
        }

    return {
        "name": name,
        "total": total,
        "correct": correct,
        "refusal_ok": refusal_ok,
        "refusal_bad": refusal_bad,
        "hallucinations": hallucinations,
        "wrong": wrong,
        "accuracy": accuracy,
        "in_scope_accuracy": in_scope_acc,
        "refusal_accuracy": refusal_acc,
        "hallucination_rate": hallucination_rate,
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "tokens_total": sum(tokens),
        "tokens_mean": sum(tokens) / len(tokens) if tokens else 0,
        "categories": cat_breakdown,
    }


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════

def _print_report(strategies: list[dict]):
    print("\n" + "=" * 80)
    print("  TOOL ROUTER BENCHMARK — RESULTS")
    print("=" * 80)

    # Summary table
    header = f"{'Strategy':<30} {'Acc':>6} {'InScope':>8} {'Refusal':>8} {'Halluc':>7} {'Lat(ms)':>8} {'Tokens':>7}"
    print(f"\n{header}")
    print("-" * len(header))
    for s in strategies:
        print(
            f"{s['name']:<30} "
            f"{s['accuracy']:>5.1%} "
            f"{s['in_scope_accuracy']:>7.1%} "
            f"{s['refusal_accuracy']:>7.1%} "
            f"{s['hallucination_rate']:>6.1%} "
            f"{s['latency_mean_ms']:>7.1f} "
            f"{s['tokens_total']:>7}"
        )

    # Per-category breakdown
    print(f"\n{'Category Accuracy':<30}", end="")
    for s in strategies:
        print(f" {s['name'][:12]:>12}", end="")
    print()
    print("-" * (30 + 13 * len(strategies)))

    all_cats = sorted({cat for s in strategies for cat in s.get("categories", {})})
    for cat in all_cats:
        print(f"  {cat:<28}", end="")
        for s in strategies:
            cat_data = s.get("categories", {}).get(cat, {})
            acc = cat_data.get("accuracy", 0)
            n = cat_data.get("total", 0)
            print(f" {acc:>5.1%} ({n:>2})", end="")
        print()

    print("\n" + "=" * 80)


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
    # Load data
    with open(QUERIES_PATH) as f:
        query_data = json.load(f)
    queries = query_data["queries"]

    with open(CATALOG_PATH) as f:
        catalog_data = json.load(f)
    tools = catalog_data["tools"]
    valid_tool_ids = {t["id"] for t in tools}

    # Determine which strategies to run
    if glyphh_only:
        active = [2]
    elif strategies:
        active = strategies
    else:
        active = [1, 2, 3, 4]

    # Check for API key if LLM strategies are requested
    llm_strategies = {1, 3, 4}
    if llm_strategies & set(active) and not os.environ.get("OPENAI_API_KEY"):
        print("⚠  OPENAI_API_KEY not set — skipping LLM strategies")
        active = [s for s in active if s not in llm_strategies]
        if not active:
            print("No strategies to run.")
            return

    # Init Glyphh router (shared across strategies 2, 3, 4)
    router = GlyphhRouter()
    print(f"Loaded {len(router.exemplar_meta)} exemplars, {len(queries)} queries, {len(tools)} tools")
    print(f"Strategies: {active}  |  LLM: {llm_model}\n")

    strategy_names = {
        1: "S1: LLM Only",
        2: "S2: Glyphh Only",
        3: "S3: LLM→Glyphh→LLM",
        4: "S4: LLM→Glyphh→ASK→LLM",
    }

    all_strategy_results = []

    for strat_num in active:
        name = strategy_names[strat_num]
        print(f"\n── {name} ──")
        results = []

        for qi, q in enumerate(queries):
            _progress(qi + 1, len(queries), name[:20])

            runs_per = LLM_RUNS_PER_QUERY if strat_num == 1 else 1
            run_results = []

            for _ in range(runs_per):
                if strat_num == 1:
                    r = strategy_1(q["query"], tools, llm_model)
                elif strat_num == 2:
                    r = strategy_2(q["query"], router)
                elif strat_num == 3:
                    r = strategy_3(q["query"], tools, router, llm_model)
                elif strat_num == 4:
                    r = strategy_4(q["query"], tools, router, llm_model)
                else:
                    continue
                run_results.append(r)

            # For S1, pick majority answer and measure consistency
            if strat_num == 1 and run_results:
                tool_ids = [r["tool_id"] for r in run_results]
                from collections import Counter
                majority = Counter(tool_ids).most_common(1)[0][0]
                consistent = all(t == tool_ids[0] for t in tool_ids)
                best = next((r for r in run_results if r["tool_id"] == majority), run_results[0])
                best["consistent"] = consistent
                best["all_answers"] = tool_ids
                run_results = [best]

            r = run_results[0]
            label = score_result(r["tool_id"], q["expected_tool"], valid_tool_ids)

            results.append({
                "query_id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "expected_tool": q["expected_tool"],
                "result_tool": r["tool_id"],
                "confidence": r.get("confidence", 0),
                "score_label": label,
                "latency_ms": r.get("latency_ms", 0),
                "tokens": r.get("tokens", 0),
                **{k: v for k, v in r.items() if k not in ("tool_id", "confidence", "latency_ms", "tokens")},
            })

        print()  # newline after progress bar
        agg = _aggregate(results, name)
        agg["raw_results"] = results
        all_strategy_results.append(agg)

    # Print comparison
    _print_report(all_strategy_results)

    # Save raw results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for agg in all_strategy_results:
            safe_name = agg["name"].replace(" ", "_").replace("→", "-").replace(":", "")
            with open(out_path / f"{safe_name}.json", "w") as f:
                json.dump(agg, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}/")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Router Benchmark: Glyphh HDC vs LLM")
    parser.add_argument("--strategies", nargs="+", type=int, choices=[1, 2, 3, 4], help="Which strategies to run (default: all)")
    parser.add_argument("--glyphh-only", action="store_true", help="Run only Strategy 2 (no API key needed)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model for LLM strategies (default: gpt-4o-mini)")
    parser.add_argument("--output", type=str, help="Directory to save raw JSON results")
    args = parser.parse_args()

    run_benchmark(
        strategies=args.strategies,
        glyphh_only=args.glyphh_only,
        llm_model=args.llm_model,
        output_dir=args.output,
    )
