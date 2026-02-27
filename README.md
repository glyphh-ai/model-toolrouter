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

# Install (single-quotes required in zsh)
pip install 'glyphh[runtime]'
```

### 2. Clone and start the model

```bash
git clone https://github.com/glyphh-ai/model-toolrouter.git
cd model-toolrouter

# Start the local dev server (no account needed)
glyphh dev .
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
Use the **up-arrow** to replay previous queries — history persists in `~/.glyphh/chat_history`.

---

## How It Works

A natural language query is decomposed into **action**, **target**, **domain**,
and **keyword** signals by the `glyphh.intent` SDK, then encoded as a
10,000-dimension bipolar vector and matched against tool exemplars via weighted
cosine similarity. The sidecar model (seed=73) independently validates any
tool references detected in the query.

```
query → assess_query() → HDC encode → cosine similarity → top tool
             ↓                          ↕ [0.40–0.55 zone]
          ASK (if                     sidecar validate
         incomplete)                  gap check → ASK
                                       (if scores too close)
```

### ASK State

The model exports `assess_query()` which the runtime calls before HDC encoding.
If the query is missing a resolvable action or domain, the runtime returns an
`ASK` response instead of a low-confidence result:

```
  ASK
  Cannot determine which service or tool to use (e.g. Slack, Jira, Stripe)
  Missing: domain
```

A second gate fires after similarity search: if the top-two scores are within
`disambiguation.min_gap` (0.03), the runtime returns ASK with disambiguation
options:

```
  ASK
  Your query matches multiple options. Did you mean one of these?
    •  slack_send_message (91% match)
    •  teams_post_message (90% match)
```

### Exemplar Design Rule

The HDC encoder produces nearly identical vectors for two exemplars that share
the same categorical role values (`action`, `target`, `domain`). Tools with
different semantics must use distinct categorical values — e.g. a "get one
ticket" tool uses `action: get, target: ticket` while a "list all customer
tickets" tool uses `action: list, target: customer`. Mixed-use exemplars
collapse the gap and cause false ASK responses.

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

53 tests · clear routing · disambiguation · adversarial · OOS abstention · ASK state

| File | What it tests |
|------|---------------|
| `test_encoding.py` | Config validation, role encoding |
| `test_similarity.py` | End-to-end routing accuracy |
| `test_states.py` | `assess_query()` completeness and ASK/DONE logic |

## Running the Full Benchmark

```bash
# Glyphh-only (no API key needed)
pytest benchmark/run.py -v -k "s2"

# All 4 strategies (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... pytest benchmark/run.py -v
```

## License

MIT
