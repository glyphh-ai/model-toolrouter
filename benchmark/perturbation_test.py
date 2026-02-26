#!/usr/bin/env python3
"""
Perturbation Test: Confidence Calibration Under Paraphrased Queries

Generates randomized paraphrases of benchmark queries and runs them
through the Glyphh router WITHOUT modifying any encoder rules. This
validates that the HDC matching (BoW similarity + lexicon signals)
generalizes beyond the exact phrasings in the benchmark.

For each in-scope query, generates 3 paraphrases using:
  1. Synonym substitution (swap action verbs + domain nouns)
  2. Word reordering (move clause order, passive voice)
  3. Combined (both synonym + reorder)

Reports:
  - Routing accuracy on paraphrased queries
  - Confidence distribution shift vs original
  - Per-category breakdown
  - Queries where paraphrase breaks routing (failure analysis)

Usage:
    python benchmark/perturbation_test.py
    python benchmark/perturbation_test.py --output benchmark/results/perturbation.json
"""

import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from run import GlyphhRouter

BENCHMARK_DIR = Path(__file__).parent
QUERIES_PATH = BENCHMARK_DIR / "queries.json"

# ---------------------------------------------------------------------------
# Synonym tables — plausible natural language substitutions.
# These must NOT appear in _PHRASE_ACTIONS or _PHRASE_TARGETS to test
# whether BoW similarity carries the match without phrase rule assistance.
# ---------------------------------------------------------------------------

_ACTION_SYNONYMS = {
    "send": ["fire off", "shoot over", "dispatch", "forward along"],
    "search": ["look through", "scan", "dig through", "hunt for"],
    "find": ["locate", "track down", "look up", "pull up"],
    "create": ["set up", "spin up", "put together", "make"],
    "get": ["grab", "fetch", "retrieve", "pull"],
    "update": ["modify", "adjust", "tweak", "change"],
    "delete": ["remove", "get rid of", "clear out", "nuke"],
    "cancel": ["terminate", "end", "discontinue", "stop"],
    "refund": ["return the money for", "reverse the charge on", "give back the payment for"],
    "track": ["log", "register", "capture", "note"],
    "upload": ["push up", "drop in", "put in", "add to"],
    "share": ["give access to", "open up to", "let someone see"],
    "list": ["show all", "display", "enumerate", "give me all"],
    "reply": ["respond to", "answer", "write back to"],
    "draft": ["prepare", "write up", "compose", "put together"],
    "set": ["configure", "put", "change to"],
    "assign": ["hand off to", "delegate to", "pass to", "give to"],
    "charge": ["bill", "invoice", "collect from"],
    "add": ["put", "leave", "write", "post"],
    "log": ["record", "note", "capture", "register"],
    "post": ["send", "drop", "fire off", "blast"],
    "look up": ["check on", "see", "retrieve", "grab"],
    "show": ["display", "pull up", "present", "give me"],
}

_NOUN_SYNONYMS = {
    "message": ["note", "notification", "ping"],
    "email": ["mail", "email message"],
    "ticket": ["issue", "task", "work item"],
    "bug": ["defect", "problem", "bug report"],
    "meeting": ["call", "sync", "session"],
    "event": ["occurrence", "activity"],
    "customer": ["client", "account holder"],
    "contact": ["person", "entry", "individual"],
    "file": ["document", "doc"],
    "channel": ["room", "channel"],
    "subscription": ["plan", "membership"],
    "invoice": ["bill", "statement"],
    "funnel": ["conversion path", "pipeline"],
    "comment": ["note", "remark", "response"],
    "status": ["availability", "state"],
    "folder": ["directory", "location"],
    "record": ["entry", "profile", "info"],
}

# Clause reordering templates — structural paraphrases
_REORDER_TEMPLATES = [
    # Move the object before the verb
    lambda q: _try_reorder_object_first(q),
    # Add filler phrases
    lambda q: _add_filler(q),
    # Passive-ish voice
    lambda q: _try_passive(q),
]


def _try_reorder_object_first(query: str) -> str:
    """Try to move a prepositional phrase to the front."""
    # "Send a message to #engineering about X" → "About X, send a message to #engineering"
    m = re.search(r'(about|regarding|for|concerning) (.+?)(?:\.|$)', query, re.IGNORECASE)
    if m:
        prep = m.group(0).rstrip('.')
        rest = query[:m.start()].strip() + query[m.end():].strip()
        return f"{prep}, {rest[0].lower()}{rest[1:]}" if rest else query
    return query


def _add_filler(query: str) -> str:
    """Add conversational filler to make the query less formulaic."""
    fillers = [
        "Can you ", "I need you to ", "Please ", "Go ahead and ",
        "I'd like to ", "Could you ", "Would you mind ",
        "Hey, ", "Quick request: ",
    ]
    filler = random.choice(fillers)
    if query[0].isupper():
        return filler + query[0].lower() + query[1:]
    return filler + query


def _try_passive(query: str) -> str:
    """Crude passive voice transformation."""
    # "Send a message to #eng" → "A message should be sent to #eng"
    patterns = [
        (r'^(Send|Post|Email|Forward)\s+(.+?)\s+(to\s+.+)$',
         lambda m: f"{m.group(2)} should be sent {m.group(3)}"),
        (r'^(Create|Make|Set up)\s+(.+?)\s+(in|for|on|about)\s+(.+)$',
         lambda m: f"{m.group(2)} needs to be created {m.group(3)} {m.group(4)}"),
        (r'^(Delete|Remove|Cancel)\s+(.+)$',
         lambda m: f"{m.group(2)} needs to be removed"),
        (r'^(Search|Find|Look)\s+(.+)$',
         lambda m: f"I'm looking for {m.group(2)}"),
    ]
    for pat, repl in patterns:
        m = re.match(pat, query, re.IGNORECASE)
        if m:
            return repl(m)
    return query


def _synonym_sub(query: str) -> str:
    """Replace one action verb and one noun with synonyms."""
    result = query
    # Try action verb substitution
    words_lower = query.lower().split()
    for word in words_lower[:5]:  # Action verbs tend to be near the start
        clean = re.sub(r'[^a-z]', '', word)
        if clean in _ACTION_SYNONYMS:
            synonyms = _ACTION_SYNONYMS[clean]
            replacement = random.choice(synonyms)
            # Replace first occurrence, case-insensitive
            result = re.sub(r'\b' + re.escape(word) + r'\b', replacement, result, count=1, flags=re.IGNORECASE)
            break

    # Try noun substitution
    for word in reversed(words_lower):  # Nouns tend to be later
        clean = re.sub(r'[^a-z]', '', word)
        if clean in _NOUN_SYNONYMS:
            synonyms = _NOUN_SYNONYMS[clean]
            replacement = random.choice(synonyms)
            result = re.sub(r'\b' + re.escape(word) + r'\b', replacement, result, count=1, flags=re.IGNORECASE)
            break

    return result


def generate_paraphrases(query: str, n: int = 3) -> list[str]:
    """Generate n paraphrased variants of a query."""
    paraphrases = []

    # Variant 1: synonym substitution
    paraphrases.append(_synonym_sub(query))

    # Variant 2: structural reorder
    reorder_fn = random.choice(_REORDER_TEMPLATES)
    reordered = reorder_fn(query)
    if reordered == query:
        # Fallback: add filler
        reordered = _add_filler(query)
    paraphrases.append(reordered)

    # Variant 3: combined
    combined = _synonym_sub(reorder_fn(query))
    if combined == query:
        combined = _synonym_sub(_add_filler(query))
    paraphrases.append(combined)

    # Ensure all variants are actually different from original
    return [p for p in paraphrases if p != query][:n]


def run_perturbation_test(output_path: str | None = None):
    """Run the perturbation test and report results."""
    with open(QUERIES_PATH) as f:
        data = json.load(f)

    queries = data["queries"]
    in_scope = [q for q in queries if q["expected_tool"] is not None]

    print(f"Loaded {len(queries)} queries, {len(in_scope)} in-scope")
    print("Generating paraphrases...")

    random.seed(42)  # Reproducible paraphrases

    router = GlyphhRouter()

    # Run original queries first for baseline
    original_results = []
    for q in in_scope:
        result = router.route(q["query"])
        original_results.append({
            "query_id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected_tool": q["expected_tool"],
            "result_tool": result["tool"],
            "confidence": result["confidence"],
            "correct": result["tool"] == q["expected_tool"],
        })

    # Run paraphrased queries
    perturbed_results = []
    all_paraphrases = []

    for q in in_scope:
        paraphrases = generate_paraphrases(q["query"])
        for i, para in enumerate(paraphrases):
            result = router.route(para)
            entry = {
                "query_id": f"{q['id']}_p{i+1}",
                "original_id": q["id"],
                "category": q["category"],
                "original_query": q["query"],
                "paraphrased_query": para,
                "expected_tool": q["expected_tool"],
                "result_tool": result["tool"],
                "confidence": result["confidence"],
                "correct": result["tool"] == q["expected_tool"],
                "original_confidence": next(
                    r["confidence"] for r in original_results if r["query_id"] == q["id"]
                ),
            }
            perturbed_results.append(entry)
            all_paraphrases.append(entry)

    # --- Report ---
    total_original = len(original_results)
    correct_original = sum(1 for r in original_results if r["correct"])
    total_perturbed = len(perturbed_results)
    correct_perturbed = sum(1 for r in perturbed_results if r["correct"])

    print()
    print("=" * 80)
    print("  PERTURBATION TEST — CONFIDENCE CALIBRATION")
    print("=" * 80)
    print()
    print(f"  Original queries:    {correct_original}/{total_original} ({correct_original/total_original:.1%})")
    print(f"  Paraphrased queries: {correct_perturbed}/{total_perturbed} ({correct_perturbed/total_perturbed:.1%})")
    print()

    # Per-category breakdown
    by_cat_orig = defaultdict(lambda: {"correct": 0, "total": 0, "confs": []})
    by_cat_pert = defaultdict(lambda: {"correct": 0, "total": 0, "confs": []})

    for r in original_results:
        by_cat_orig[r["category"]]["correct"] += int(r["correct"])
        by_cat_orig[r["category"]]["total"] += 1
        by_cat_orig[r["category"]]["confs"].append(r["confidence"])

    for r in perturbed_results:
        by_cat_pert[r["category"]]["correct"] += int(r["correct"])
        by_cat_pert[r["category"]]["total"] += 1
        by_cat_pert[r["category"]]["confs"].append(r["confidence"])

    print(f"  {'Category':<20s}  {'Original':>10s}  {'Perturbed':>10s}  {'Conf Orig':>10s}  {'Conf Pert':>10s}  {'Shift':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for cat in ["clear", "near_collision", "adversarial", "schema_trap"]:
        if cat not in by_cat_orig:
            continue
        o = by_cat_orig[cat]
        p = by_cat_pert[cat]
        o_acc = f"{o['correct']}/{o['total']}"
        p_acc = f"{p['correct']}/{p['total']}"
        o_avg = sum(o["confs"]) / len(o["confs"]) if o["confs"] else 0
        p_avg = sum(p["confs"]) / len(p["confs"]) if p["confs"] else 0
        shift = p_avg - o_avg
        print(f"  {cat:<20s}  {o_acc:>10s}  {p_acc:>10s}  {o_avg:>10.4f}  {p_avg:>10.4f}  {shift:>+8.4f}")

    # Confidence zone distribution
    print()
    print("  CONFIDENCE ZONES (paraphrased queries)")
    zones = {"high (>=0.55)": 0, "uncertain (0.40-0.55)": 0, "abstain (<0.40)": 0}
    for r in perturbed_results:
        if r["confidence"] >= 0.55:
            zones["high (>=0.55)"] += 1
        elif r["confidence"] >= 0.40:
            zones["uncertain (0.40-0.55)"] += 1
        else:
            zones["abstain (<0.40)"] += 1
    for zone, count in zones.items():
        pct = count / total_perturbed if total_perturbed > 0 else 0
        print(f"    {zone}: {count}/{total_perturbed} ({pct:.0%})")

    # Failure analysis
    failures = [r for r in perturbed_results if not r["correct"]]
    if failures:
        print()
        print(f"  FAILURES ({len(failures)} paraphrased queries routed incorrectly)")
        print()
        for f in failures[:20]:
            print(f"    [{f['query_id']}] expected={f['expected_tool']}, got={f['result_tool']} (conf={f['confidence']:.4f})")
            print(f"      Original:    {f['original_query'][:70]}")
            print(f"      Paraphrased: {f['paraphrased_query'][:70]}")
            print()
    else:
        print()
        print("  No failures — all paraphrased queries routed correctly.")

    print("=" * 80)

    return {
        "original_results": original_results,
        "perturbed_results": perturbed_results,
        "all_paraphrases": all_paraphrases,
        "in_scope": in_scope,
        "original_accuracy": correct_original / total_original,
        "perturbed_accuracy": correct_perturbed / total_perturbed,
    }


def run_llm_perturbation(all_paraphrases: list[dict], in_scope: list[dict],
                         model: str = "gpt-4o-mini"):
    """Run S1 (LLM Only) on the same paraphrased queries for comparison."""
    from run import strategy_1, CATALOG_PATH

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    tools = catalog["tools"]

    # Run LLM on original queries first
    print()
    print("=" * 80)
    print(f"  LLM PERTURBATION TEST ({model})")
    print("=" * 80)
    print()

    llm_original = []
    total = len(in_scope)
    for i, q in enumerate(in_scope):
        sys.stdout.write(f"\r  LLM originals [{i+1}/{total}]")
        sys.stdout.flush()
        result = strategy_1(q["query"], tools, model)
        llm_original.append({
            "query_id": q["id"],
            "category": q["category"],
            "expected_tool": q["expected_tool"],
            "result_tool": result["tool"],
            "correct": result["tool"] == q["expected_tool"],
        })
    print()

    # Run LLM on paraphrased queries
    llm_perturbed = []
    total = len(all_paraphrases)
    for i, p in enumerate(all_paraphrases):
        sys.stdout.write(f"\r  LLM paraphrased [{i+1}/{total}]")
        sys.stdout.flush()
        result = strategy_1(p["paraphrased_query"], tools, model)
        llm_perturbed.append({
            "query_id": p["query_id"],
            "original_id": p["original_id"],
            "category": p["category"],
            "paraphrased_query": p["paraphrased_query"],
            "expected_tool": p["expected_tool"],
            "result_tool": result["tool"],
            "correct": result["tool"] == p["expected_tool"],
        })
    print()

    llm_orig_correct = sum(1 for r in llm_original if r["correct"])
    llm_pert_correct = sum(1 for r in llm_perturbed if r["correct"])

    print()
    print(f"  LLM original:    {llm_orig_correct}/{len(llm_original)} ({llm_orig_correct/len(llm_original):.1%})")
    print(f"  LLM paraphrased: {llm_pert_correct}/{len(llm_perturbed)} ({llm_pert_correct/len(llm_perturbed):.1%})")
    print()

    # Per-category
    by_cat_orig = defaultdict(lambda: {"correct": 0, "total": 0})
    by_cat_pert = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in llm_original:
        by_cat_orig[r["category"]]["correct"] += int(r["correct"])
        by_cat_orig[r["category"]]["total"] += 1
    for r in llm_perturbed:
        by_cat_pert[r["category"]]["correct"] += int(r["correct"])
        by_cat_pert[r["category"]]["total"] += 1

    print(f"  {'Category':<20s}  {'LLM Orig':>10s}  {'LLM Pert':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
    for cat in ["clear", "near_collision", "adversarial", "schema_trap"]:
        if cat not in by_cat_orig:
            continue
        o = by_cat_orig[cat]
        p = by_cat_pert[cat]
        print(f"  {cat:<20s}  {o['correct']:>3d}/{o['total']:<4d}   {p['correct']:>3d}/{p['total']:<4d}")

    # LLM failures on paraphrased
    llm_failures = [r for r in llm_perturbed if not r["correct"]]
    if llm_failures:
        print()
        print(f"  LLM FAILURES ({len(llm_failures)} paraphrased queries)")
        for f in llm_failures[:15]:
            print(f"    [{f['query_id']}] expected={f['expected_tool']}, got={f['result_tool']}")
            print(f"      {f['paraphrased_query'][:75]}")

    print()
    print("=" * 80)

    return {
        "llm_original": llm_original,
        "llm_perturbed": llm_perturbed,
        "llm_orig_accuracy": llm_orig_correct / len(llm_original),
        "llm_pert_accuracy": llm_pert_correct / len(llm_perturbed),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Save raw results to JSON")
    parser.add_argument("--llm", action="store_true", help="Also run S1 (LLM) on same paraphrases")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model for --llm")
    args = parser.parse_args()

    glyphh_data = run_perturbation_test(args.output)

    llm_data = {}
    if args.llm:
        llm_data = run_llm_perturbation(
            glyphh_data["all_paraphrases"],
            glyphh_data["in_scope"],
            model=args.llm_model,
        )

    # Summary comparison
    if args.llm:
        print()
        print("=" * 80)
        print("  SIDE-BY-SIDE: Glyphh vs LLM on Paraphrased Queries")
        print("=" * 80)
        print()
        print(f"  {'':30s}  {'Original':>10s}  {'Paraphrased':>12s}  {'Delta':>8s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*12}  {'-'*8}")
        g_orig = glyphh_data["original_accuracy"]
        g_pert = glyphh_data["perturbed_accuracy"]
        l_orig = llm_data["llm_orig_accuracy"]
        l_pert = llm_data["llm_pert_accuracy"]
        print(f"  {'Glyphh HDC (S2)':<30s}  {g_orig:>9.1%}  {g_pert:>11.1%}  {g_pert-g_orig:>+7.1%}")
        print(f"  {'LLM gpt-4o-mini (S1)':<30s}  {l_orig:>9.1%}  {l_pert:>11.1%}  {l_pert-l_orig:>+7.1%}")
        print()
        print("=" * 80)

    # Save combined results
    if args.output:
        output = {
            "test": "perturbation",
            "seed": 42,
            "glyphh_original_accuracy": glyphh_data["original_accuracy"],
            "glyphh_perturbed_accuracy": glyphh_data["perturbed_accuracy"],
            "total_original": len(glyphh_data["original_results"]),
            "total_perturbed": len(glyphh_data["perturbed_results"]),
            "original_results": glyphh_data["original_results"],
            "perturbed_results": glyphh_data["perturbed_results"],
        }
        if llm_data:
            output["llm_original_accuracy"] = llm_data["llm_orig_accuracy"]
            output["llm_perturbed_accuracy"] = llm_data["llm_pert_accuracy"]
            output["llm_original"] = llm_data["llm_original"]
            output["llm_perturbed"] = llm_data["llm_perturbed"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
