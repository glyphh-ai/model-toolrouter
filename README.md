# SaaS Tool Router

Routes natural language SaaS requests to the correct tool function using HDC
similarity matching on intent vectors. 38 tools across 8 domains — Slack,
email, CRM, Stripe, calendar, Google Drive, Jira, and analytics.

## How It Works

Describe what you want in plain English — "refund charge ch_abc123", "create a
bug ticket in ENG", "find a time for a 30-minute call" — and the encoder
decomposes your query into action, target, domain, and keyword signals, encodes
them as high-dimensional vectors, and matches against exemplars via cosine
similarity. No LLM required for routing. Deterministic. Sub-10ms.

## Benchmark Results

> **Internal Glyphh Eval — Formal Benchmark Validation Pending**

95 queries, 38 tools, 5 difficulty categories (clear, near-collision,
adversarial, open-set, schema-trap). Four strategies compared:

| Strategy | What it does | Routing Acc | In-Scope | Tokens | Latency |
|---|---|---|---|---|---|
| LLM Only | LLM picks tool + args | 91.6% | 88.7% | 229K | 1,820 ms |
| Glyphh Only | HDC routes, no args | 100% | 100% | 0 | 6 ms |
| Glyphh Route → LLM Args | HDC routes, LLM fills args | 100% | 100% | 22K | 658 ms |
| LLM + Glyphh Sidecar | LLM does everything, HDC confirms/overrides | 98.9% | 100% | 232K | 1,655 ms |

**Glyphh Route → LLM Args** delivers 100% accuracy with 90% fewer tokens and
64% lower latency than LLM-only. The LLM never sees the full catalog — it
receives one tool definition and extracts args. This is the architecture for
cost-sensitive, high-volume deployments.

**LLM + Glyphh Sidecar** is the drop-in pattern for existing LLM pipelines.
Keep your LLM, bolt Glyphh on as a sidecar. It catches and corrects every
in-scope routing error the LLM makes — 100% in-scope accuracy with zero
changes to your LLM prompt or flow.

Full analysis: [`benchmark/analysis.md`](benchmark/analysis.md)

## Model Structure

```
toolrouter/
├── config.yaml              # model config, roles, similarity settings
├── encoder.py               # EncoderConfig + NL query encoder
├── data/
│   └── exemplars.jsonl      # 68 tool exemplars across 38 tools
├── benchmark/
│   ├── run.py               # 4-strategy benchmark runner
│   ├── queries.json         # 95 test queries (v2.2.0)
│   ├── tool_catalog.json    # 38 tool definitions with schemas
│   ├── analysis.md          # detailed findings
│   └── results/             # raw JSON results per strategy
├── tests/                   # unit tests
├── build.py                 # package model into .glyphh file
└── manifest.yaml            # model identity
```

## Encoder Architecture

Two-layer HDC encoder (10,000 dimensions):

| Layer | Weight | Segments | Purpose |
|---|---|---|---|
| intent | 0.7 | action (verb + target), scope (domain) | Primary routing signal |
| context | 0.3 | keywords | Disambiguation via keyword overlap |

## Domains

Messaging (Slack), Email (Gmail), CRM, Payments (Stripe), Calendar, Files
(Google Drive), Tickets (Jira), Analytics.

## Running the Benchmark

```bash
# Glyphh-only (no API key needed)
./dev-models.sh benchmark toolrouter

# All 4 strategies (requires OPENAI_API_KEY)
export $(grep OPENAI_API_KEY .env | xargs)
./dev-models.sh benchmark toolrouter --strategies 1 2 3 4 \
  --output glyphh-models/toolrouter/benchmark/results/
```
