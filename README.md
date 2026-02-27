# SaaS Tool Router

Routes natural language SaaS requests to the correct tool function using
Hyperdimensional Computing (HDC) similarity matching. 39 tools across 8
domains — Slack, Email, CRM, Stripe, Calendar, Google Drive, Jira, Analytics.

No LLM required for routing. Deterministic. Sub-10ms.

**[Docs →](https://glyphh.ai/docs)** · **[Glyphh Hub →](https://glyphh.ai/hub)**

---

## Getting Started

### 1. Install the Glyphh CLI

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install
pip install glyphh
```

### 2. Clone and start the model

```bash
git clone https://github.com/glyphh-ai/model-toolrouter.git
cd model-toolrouter

# Start the local dev server (--debug flag streams logs)
glyphh dev . -d
```

The server starts at `http://localhost:8002`. Open the Chat UI in your browser
at the URL printed in the terminal.

### 3. Query the model

```bash
# Single query
glyphh chat "Refund charge ch_3abc123 for the full amount"

# Interactive REPL
glyphh chat
```

Example output:

```
  DONE
   94%  [███████████░]  stripe_refund
   61%  [███████░░░░░]  stripe_get_customer

  8.2ms · auto · similarity_search
```

Switch to GQL mode inside the REPL with `/gql`, back with `/nl`, exit with `/quit`.

---

## How It Works

A natural language query is decomposed into **action**, **target**, **domain**,
and **keyword** signals by the `glyphh.intent` SDK, then encoded as a
10,000-dimension bipolar vector and matched against tool exemplars via weighted
cosine similarity. The sidecar model (seed=73) independently validates any
tool references detected in the query.

```
query → IntentExtractor → HDC encode → cosine similarity → top tool
                                       ↕ [0.40–0.55 zone]
                                     sidecar validate
```

## Benchmark

95 queries · 39 tools · 5 difficulty categories

| Strategy | Routing | In-Scope E2E | Tokens/query | Latency |
|---|---|---|---|---|
| LLM Only | 91.6% | 88.7% | 2,416 | 1,749 ms |
| **Glyphh Only** | **98.9%** | — | **0** | **7 ms** |
| **Glyphh Route → LLM Args** | **100%** | **100%** | **228** | **626 ms** |
| LLM + Glyphh Sidecar | 100% | 100% | 2,440 | 1,916 ms |

**Glyphh Route → LLM Args** is the recommended production pattern: HDC picks
the tool deterministically, the LLM fills args against a single tool schema.
90% token reduction, 64% lower latency vs LLM-only, 100% accuracy.

Full benchmark details: [`benchmark/analysis.md`](benchmark/analysis.md)

## Encoder

Two-layer HDC encoder (10,000 dimensions, `glyphh.intent` for NL extraction):

| Layer | Weight | Segments | Signal |
|---|---|---|---|
| intent | 0.4 | action (verb + target), scope (domain) | Primary routing |
| semantics | 0.6 | description + keywords (BoW) | Fuzzy disambiguation |

## Running Tests

```bash
cd model-toolrouter
pip install -r requirements-dev.txt
pytest tests/ -v
```

40 tests · clear routing · disambiguation · adversarial · OOS abstention

## Running the Full Benchmark

```bash
# Glyphh-only (no API key needed)
pytest benchmark/run.py -v -k "s2"

# All 4 strategies (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... pytest benchmark/run.py -v
```

## License

MIT
