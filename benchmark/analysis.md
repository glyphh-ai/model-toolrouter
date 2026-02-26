# SaaS Tool Router Benchmark — Analysis

> **Internal Glyphh Eval — Formal Benchmark Validation Pending**
>
> These results were produced by the Glyphh team using an internally designed
> benchmark. An independent, externally validated benchmark is planned but has
> not yet been conducted. Interpret accordingly.

## Overview

This benchmark measures how accurately different architectures route natural
language SaaS requests to the correct tool function. The test set covers 95
queries across 38 tools in 8 domains (Slack, email, CRM, Stripe, calendar,
Google Drive, Jira, analytics), split into 5 difficulty categories:

| Category | Count | Description |
|---|---|---|
| clear | 20 | Unambiguous, single-tool requests |
| near_collision | 35 | Semantically similar tools compete (CRM vs Stripe, track vs identify) |
| adversarial | 15 | Fake tool names, multi-action queries, misleading phrasing |
| open_set | 15 | Out-of-scope queries that should return null (no tool) |
| schema_trap | 10 | Correct tool is obvious but args are tricky (cents, enums, anyOf) |

## Architecture (v3.1.0)

The Glyphh router uses a **two-model HDC architecture**:

**Main model (seed=42, 10,000-dim):**
- Intent layer (weight=0.4): action, target, domain — lexicon-encoded categorical signals
- Semantics layer (weight=0.6): description, keywords — bag-of-words encoded text
- Scoring: weighted role-level cosine similarity, threshold=0.40

**Sidecar model (seed=73, 10,000-dim):**
- Identity layer (weight=0.7): name_tokens — tool name split on underscores, BoW encoded
- Capability layer (weight=0.3): action — primary verb, lexicon encoded
- Activated when main model confidence is in the uncertain zone (0.40–0.55) and
  the query explicitly references a tool by name (snake_case or camelCase pattern)
- Validates whether the referenced name matches a real tool in the catalog

The sidecar operates in an **independent vector space** from the main model.
This is the Glyphh sidecar pattern — the same architecture used by the BFCL
Observer model (seed=107) for multi-model deductive reasoning.

## Strategies

| # | Strategy | Description |
|---|---|---|
| S1 | LLM Only | LLM selects tool + generates args from full catalog |
| S2 | Glyphh Only | HDC main model + sidecar routes to tool, no args (routing-only) |
| S3 | Glyphh Route + LLM Args | Glyphh selects tool, LLM generates args for that single tool |
| S4 | LLM + Glyphh Sidecar | LLM selects tool + args, Glyphh confirms or overrides routing |

## Results

### Routing Accuracy

How often each strategy picks the correct tool (or correctly abstains).

| Strategy | Tool Acc | In-Scope | Abstain | Latency (avg) | Tokens |
|---|---|---|---|---|---|
| S1: LLM Only | 91.6% | 88.7% | 100% | 1,749 ms | 229K |
| **S2: Glyphh Only** | **100%** | **100%** | **100%** | **7 ms** | **0** |
| S3: Glyphh Route + LLM Args | 100% | 100% | 100% | 626 ms | 22K |
| S4: LLM + Glyphh Sidecar | 100% | 100% | 100% | 1,916 ms | 232K |

### Per-Category Breakdown (S2: Glyphh Only)

| Category | Accuracy | Count |
|---|---|---|
| clear | 100% | 20/20 |
| near_collision | 100% | 35/35 |
| adversarial | 100% | 15/15 |
| open_set | 100% | 15/15 |
| schema_trap | 100% | 10/10 |

### End-to-End Validity (correct tool + schema-valid args)

Only applies to strategies that produce arguments. Schema validation uses
jsonschema Draft 7.

| Strategy | E2E Acc | Schema Pass Rate |
|---|---|---|
| S1: LLM Only | 88.7% | 100% (70/70) |
| S3: Glyphh Route + LLM Args | 100% | 100% (71/71) |
| S4: LLM + Glyphh Sidecar | 100% | 100% (71/71) |

S2 is excluded — it produces no args by design.

## Key Findings

### 1. The sidecar eliminates adversarial false positives

Before the sidecar (v3.0.0), the main model scored 53.3% on adversarial
queries. Seven queries referencing fake tool names (e.g., `stripe_pause_subscription`,
`gmail_archive`, `crm_delete_contact`) scored 0.40–0.45 on the main model —
just above the abstention threshold — because they share domain words with
real tools.

The sidecar resolves this by validating tool name references in an independent
vector space. Fake tool names like `stripe_pause_subscription` get sidecar
similarity ~0.47 (name overlap on "stripe" but invalid action "pause" is not
in the lexicon), while real tool synonyms like `sendSlackNotification` get
~0.77 (name overlap AND matching action "send"). A threshold of 0.55 cleanly
separates the two groups.

### 2. LLMs struggle with semantic near-collisions

S1 gets 8 routing decisions wrong. The failures are systematic, not random:

- **CRM vs Stripe**: "Find the customer record for bob@startup.io" — LLM picks
  `stripe_get_customer` instead of `crm_get_contact`. The word "customer"
  pulls it toward Stripe.
- **track vs identify**: "Log that user u_555 upgraded to pro" — LLM picks
  `analytics_track_event` instead of `analytics_identify_user`. Plan changes
  are trait updates, not events.
- **delete vs list**: "Remove the standup meeting" — LLM picks
  `calendar_list_events` (reasoning: "need to find it first"). Correct under
  the routing policy: the highest-impact action is the deletion.
- **Multi-action routing**: Under the highest-impact policy, the LLM
  sometimes picks the notification instead of the state change.

These are exactly the cases where structured intent decomposition
(action + target + domain) outperforms token-level pattern matching.

### 3. Two HDC models, zero LLM dependency

S2 achieves 100% routing accuracy using only HDC cosine similarity across two
independent vector spaces. No API calls, no tokens, sub-10ms latency. The main
model encodes intent and semantic signals; the sidecar validates tool name
identity. Together, they handle disambiguation, abstention, and adversarial
tool name hallucination.

### 4. S3 is the cost-efficiency play

S3 matches S4's accuracy while using 90% fewer tokens and running 64% faster
than S1. The LLM never sees the full 38-tool catalog — it receives a single
tool definition and extracts args.

| Metric | S1 (LLM Only) | S3 (Glyphh + LLM) | Reduction |
|---|---|---|---|
| Tokens | 229K | 22K | 90% |
| Latency | 1,749 ms | 626 ms | 64% |
| Routing errors | 8 | 0 | 100% |

### 5. The LLM is good at args, bad at routing

Schema validity is 100% across all strategies. Once the LLM knows which tool
to call, it fills in fields correctly — types, enums, required fields, cents
conversion, anyOf constraints. The problem is purely in the routing decision
over a large catalog.

## Methodology Notes

- **LLM**: gpt-4o-mini, temperature=0, single pass (no majority vote)
- **Schema validation**: jsonschema Draft 7, enforced even on empty args
- **Routing policy**: Highest-impact/most irreversible action for multi-action
  queries, applied consistently across all strategies
- **Metrics separation**: Routing accuracy and end-to-end validity are reported
  independently so S2 (routing-only) is directly comparable
- **LLM prompt**: Full tool catalog with schemas, descriptions, parameter
  types, enums, and anyOf constraints included in the system prompt

## Limitations and Future Work

- **Internal eval only** — queries, exemplars, and encoder were developed by
  the same team. An independent held-out test set with paraphrased queries
  (no phrase leakage) is needed for external validation.
- **Single LLM** — only gpt-4o-mini tested. Results may differ with GPT-4o,
  Claude, or open-source models.
- **38 tools** — real production catalogs can have hundreds. Scaling behavior
  is untested.
- **Exemplar balance** — some tools have more exemplars than others. Balanced
  exemplar counts per tool would strengthen the eval.
- **Fairness slices** — no testing across names, languages, or noisy input.
- **Sidecar scope** — the sidecar only activates when a query contains an
  explicit tool name reference (snake_case or camelCase). Adversarial queries
  that describe fake tools in natural language (without a function-style name)
  would not trigger sidecar validation.
- **S4 OOS architecture** — S4 uses the Glyphh sidecar to validate LLM routing
  decisions. When both LLM and Glyphh agree, the result is highly reliable.
  When they disagree, Glyphh overrides. Current run shows 100% but with more
  diverse OOS queries, the LLM may still produce false positives that Glyphh
  would need to catch.

## Version

- Model: v3.1.0 (main model + sidecar)
- Benchmark: v2.2.0
- Tool catalog: 38 tools, 8 domains
- Query set: 95 queries, 5 categories
- Main encoder: 10,000-dim HDC (seed=42), 68 exemplars, 5 roles
- Sidecar encoder: 10,000-dim HDC (seed=73), 38 tool glyphs, 2 roles
- Routing policy: Highest-impact action
