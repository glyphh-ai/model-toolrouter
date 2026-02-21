# Tool Router Benchmark: Glyphh HDC vs LLM

Compares four tool routing strategies on the same query set and tool catalog.

## Strategies

| # | Name | Flow | LLM Calls |
|---|------|------|-----------|
| 1 | LLM Only | query → LLM → result | 1 |
| 2 | Glyphh Only | query → Glyphh HDC → result | 0 |
| 3 | LLM → Glyphh → LLM | query → LLM extracts intent → Glyphh routes → LLM confirms | 2 |
| 4 | LLM → Glyphh → ASK → LLM → Glyphh → LLM | Same as 3, but if Glyphh confidence is low: LLM asks clarifying question → simulated answer → re-extract → re-route → confirm | 2–6 |

## What It Measures

| Metric | Description |
|--------|-------------|
| Accuracy | Correct tool selected for the query |
| Hallucination Rate | Tool selected that doesn't exist in the catalog |
| Refusal Accuracy | Correctly returns "no match" for out-of-scope queries |
| Consistency | LLM gives same answer across 3 runs (strategy 1 only) |
| Latency | Time per query (ms) |
| Cost | Estimated token cost per query |

## Query Categories (60 queries)

| Category | Count | Purpose |
|----------|-------|---------|
| clear | 12 | Unambiguous queries with obvious tool matches |
| ambiguous | 8 | Queries that could match multiple tools |
| paraphrased | 12 | Same intent expressed in unusual phrasing |
| out_of_scope | 10 | Queries that should NOT match any tool |
| adversarial | 10 | Fake tools, misleading keywords, unsupported params |

## Running

```bash
# All four strategies
python benchmark/run.py

# Glyphh only (no API key needed)
python benchmark/run.py --glyphh-only

# Specific strategies
python benchmark/run.py --strategies 1 2
python benchmark/run.py --strategies 2 3 4

# Different LLM
python benchmark/run.py --llm-model gpt-4o

# Save raw results for verification
python benchmark/run.py --output benchmark/results/
```

## Methodology

- Both systems receive identical tool catalogs (17 tools) and queries (60)
- LLM gets the catalog as a system prompt with structured JSON output
- Glyphh encodes the same catalog as HDC exemplars from `data/exemplars.jsonl`
- Strategy 1 runs each query 3x through the LLM to measure consistency
- Strategy 4 only triggers the ASK loop when Glyphh confidence < 0.55
- Ground truth labels are human-authored, not LLM-generated
- Raw results saved as JSON for independent verification
- All LLM calls use temperature=0.0 for reproducibility
