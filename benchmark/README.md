# Tool Router Benchmark: Glyphh HDC vs LLM

Confidence-based tool routing for multi-turn agent architectures.
Compares four strategies on a 95-query, 38-tool SaaS catalog.

## How It Works

The Glyphh router doesn't replace the LLM — it replaces the LLM's
**reasoning about which tool to call**. The model decomposes a natural
language query into structured intent signals (action, target, domain,
keywords), encodes them as HDC vectors, and returns a match with a
calibrated confidence score and fact tree.

The downstream agent uses the confidence to decide:

| Zone | Confidence | Agent Action |
|---|---|---|
| High | >= 0.55 | Execute — route is reliable |
| Uncertain | 0.40–0.55 | Clarify — inspect fact tree, ask user |
| Abstain | < 0.40 | No match — out of scope |

The LLM never sees the full tool catalog. It only sees the single
routed tool (for arg extraction) or the fact tree (for clarification).

## Strategies

| # | Name | Flow | LLM Calls |
|---|------|------|-----------|
| 1 | LLM Only | query → LLM → tool + args | 1 |
| 2 | Glyphh Only | query → main model + sidecar → tool + confidence | 0 |
| 3 | Glyphh Route + LLM Args | query → Glyphh routes → LLM fills args for single tool | 1 |
| 4 | LLM + Glyphh Sidecar | query → LLM → tool + args, Glyphh confirms or overrides | 1–2 |

## Architecture

Two-model HDC architecture in independent vector spaces:

- **Main model** (seed=42, 10,000-dim): intent lexicons + semantic BoW
  for routing. Decomposes query into action/target/domain (categorical) +
  description/keywords (bag-of-words), matches against 68 exemplar glyphs.

- **Sidecar model** (seed=73, 10,000-dim): tool name BoW + action lexicon
  for adversarial validation. Activates in the uncertain zone (0.40–0.55)
  when the query references a tool by name. Validates whether the name
  matches a real tool in the catalog.

## What It Measures

| Metric | Description |
|--------|-------------|
| Tool Accuracy | Correct tool selected for the query |
| In-Scope Accuracy | Correct tool for queries that have a real tool match |
| Abstain Accuracy | Correctly returns null for out-of-scope queries |
| Confidence Distribution | Score distribution across confidence zones |
| Invalid Tool Rate | Tool selected that doesn't exist in the catalog |
| Schema Validity | Args match the tool's JSON Schema (Draft 7) |
| E2E Accuracy | Correct tool AND schema-valid args |
| Latency | Time per query (ms) |
| Tokens | Total LLM token usage |

## Query Categories (95 queries)

| Category | Count | Purpose |
|----------|-------|---------|
| clear | 20 | Unambiguous queries with obvious tool matches |
| near_collision | 35 | Semantically similar tools compete |
| adversarial | 15 | Fake tool names, multi-action, misleading phrasing |
| open_set | 15 | Queries that should NOT match any tool |
| schema_trap | 10 | Correct tool but tricky args (cents, enums, anyOf) |

## Running

```bash
# Glyphh only (no API key needed)
python benchmark/run.py --glyphh-only

# All four strategies (needs OPENAI_API_KEY)
python benchmark/run.py --strategies 1 2 3 4

# Specific strategies
python benchmark/run.py --strategies 2 3

# Different LLM
python benchmark/run.py --llm-model gpt-4o

# Save raw results for verification
python benchmark/run.py --glyphh-only --output benchmark/results/
python benchmark/run.py --strategies 1 2 3 4 --output benchmark/results/
```

Or from the monorepo root:

```bash
./dev-models.sh benchmark toolrouter
./dev-models.sh benchmark toolrouter --strategies 1 2 3 4
```

## Perturbation Testing

The perturbation test validates confidence calibration on unseen phrasings.
Each in-scope query is paraphrased 3 ways (synonym substitution, word
reordering, conversational filler) and run through the router without
modification.

```bash
python benchmark/perturbation_test.py
python benchmark/perturbation_test.py --llm  # Compare against LLM
```

Current results (206 paraphrased queries):

| Router | Original | Paraphrased | Delta |
|---|---|---|---|
| Glyphh HDC (S2) | 100% | 83.5% | -16.5% |
| LLM gpt-4o-mini (S1) | 90.1% | 88.8% | -1.3% |

Of the 34 remaining failures, only 2 are high-confidence wrong answers.
The rest are either correct abstentions (BoW too different) or uncertain-zone
near-collisions where the multi-turn agent would clarify via fact tree.

## Roadmap: Shared Intent Model

The NL extraction layer in this model (action verb maps, noun synonyms,
domain signals, phrase disambiguation) is **not specific to tool routing**.
These are universal patterns that any Glyphh model consuming natural language
needs: "fire off" means "send" whether you're routing SaaS tools, classifying
support tickets, or predicting churn.

The next step is to extract this into a dedicated **intent model**
(`glyphh-models/intent/`) that:

1. Consolidates all verb/noun/domain extraction into a shared, importable module
2. Uses HDC encoding for synonym matching — unknown verbs matched by vector
   similarity to known verb clusters, eliminating the need for exhaustive lexicons
3. Supports lifelong learning — `extractor.learn(query, correct_action)` adds
   new mappings that all downstream models benefit from immediately
4. Enables the perturbation → failure → learn → validate cycle to run
   automatically in CI

Domain models (toolrouter, BFCL, future models) would import the shared intent
layer and only define their own catalog-specific exemplars and thresholds.

## Methodology

- 38 tools across 8 domains (Slack, email, CRM, Stripe, calendar, Drive, Jira, analytics)
- 68 exemplars encoded as 10,000-dimensional HDC vectors
- LLM: gpt-4o-mini, temperature=0.0, single pass
- Schema validation: jsonschema Draft 7, enforced even on empty args
- Routing policy: highest-impact action for multi-action queries
- Ground truth labels are human-authored, not LLM-generated
- Per-query confidence scores saved in raw JSON results
- Raw results saved as JSON for independent verification
