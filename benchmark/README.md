# Tool Router Benchmark: Glyphh HDC vs LLM

Compares four tool routing strategies on the same 95-query, 38-tool SaaS catalog.

## Strategies

| # | Name | Flow | LLM Calls |
|---|------|------|-----------|
| 1 | LLM Only | query → LLM → tool + args | 1 |
| 2 | Glyphh Only | query → main model + sidecar → tool (no args) | 0 |
| 3 | Glyphh Route + LLM Args | query → Glyphh routes → LLM fills args for single tool | 1 |
| 4 | LLM + Glyphh Sidecar | query → LLM → tool + args, Glyphh confirms or overrides | 1–2 |

## Architecture

S2 uses a two-model HDC architecture:
- **Main model** (seed=42): intent lexicons + semantic BoW for routing
- **Sidecar model** (seed=73): tool name BoW + action lexicon for adversarial validation

The sidecar activates when the main model is uncertain (confidence 0.40–0.55) and the query references a tool by name (snake_case or camelCase pattern).

## What It Measures

| Metric | Description |
|--------|-------------|
| Tool Accuracy | Correct tool selected for the query |
| In-Scope Accuracy | Correct tool for queries that have a real tool match |
| Abstain Accuracy | Correctly returns null for out-of-scope queries |
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

## Methodology

- 38 tools across 8 domains (Slack, email, CRM, Stripe, calendar, Drive, Jira, analytics)
- 68 exemplars encoded as 10,000-dimensional HDC vectors
- LLM: gpt-4o-mini, temperature=0.0, single pass
- Schema validation: jsonschema Draft 7, enforced even on empty args
- Routing policy: highest-impact action for multi-action queries
- Ground truth labels are human-authored, not LLM-generated
- Raw results saved as JSON for independent verification
