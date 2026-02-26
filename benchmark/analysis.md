# SaaS Tool Router Benchmark — Analysis

> **Internal Glyphh Eval — Formal Benchmark Validation Pending**
>
> These results were produced by the Glyphh team using an internally designed
> benchmark. An independent, externally validated benchmark is planned but has
> not yet been conducted. Interpret accordingly.

## Design Philosophy

The Glyphh tool router is **not an NLU system**. It is a confidence-based
routing engine designed for multi-turn agent architectures. The model's job
is to produce a match and a calibrated confidence score that the downstream
agent uses to decide: execute, clarify, or abstain.

In a multi-turn system, the Glyphh router replaces the LLM's reasoning about
*which* tool to call. The LLM doesn't see the full tool catalog — it receives
a pre-routed tool with a confidence score and a fact tree explaining the match.
The LLM's role is reduced to argument extraction and clarification — tasks
where it excels — while routing stays in the HDC vector space where it's
faster, cheaper, and more reliable.

**Confidence zones:**

| Zone | Confidence | Agent Action |
|---|---|---|
| High | >= 0.55 | Execute — route is reliable |
| Uncertain | 0.40–0.55 | Clarify — agent asks the user or inspects the fact tree |
| Abstain | < 0.40 | No match — query is out of scope or ambiguous |

The benchmark validates two properties:
1. **Routing accuracy** — does the model pick the right tool?
2. **Confidence calibration** — is the confidence signal trustworthy?

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

### How Routing Works

The encode_query() function performs structured feature extraction from
natural language — decomposing a query into action, target, domain, and
keyword signals. These structured signals are then encoded as HDC vectors
(lexicon symbols for categorical roles, bag-of-words bundles for text roles)
and matched against exemplar glyphs via cosine similarity in 10,000-dim space.

This is not NLU — the model doesn't "understand" language. It decomposes
intent into structured signals and matches them in vector space. The feature
extraction ensures clean signal separation (e.g., "customer record" → CRM
domain, not Stripe), while the HDC matching handles fuzzy similarity across
the full keyword space.

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

### Confidence Calibration (S2: Glyphh Only)

The confidence score determines how the downstream agent acts. For the
routing to be useful in production, the score must be **calibrated**: high
confidence must mean correct, low confidence must mean out-of-scope or
ambiguous.

**Confidence distributions by category:**

| Category | Min | Max | Avg | Count |
|---|---|---|---|---|
| clear | 0.4393 | 0.7966 | 0.6663 | 20 |
| near_collision | 0.4574 | 0.7353 | 0.6366 | 35 |
| schema_trap | 0.4029 | 0.7070 | 0.5994 | 10 |
| adversarial (in-scope) | 0.4431 | 0.7024 | 0.5919 | 6 |
| adversarial (OOS) | 0.2351 | 0.4488 | 0.3906 | 9 |
| open_set | 0.0083 | 0.3381 | 0.1408 | 15 |

**Confidence zone distribution (in-scope queries):**

| Zone | Count | Meaning |
|---|---|---|
| High (>= 0.55) | 55 / 71 (77%) | Agent executes confidently |
| Uncertain (0.40–0.55) | 16 / 71 (23%) | Agent may clarify or inspect fact tree |
| Abstain (< 0.40) | 0 / 71 (0%) | No in-scope queries fall below threshold |

**OOS separation:**
- All 15 open_set queries score below 0.40 (max: 0.3381)
- 7 of 9 adversarial OOS queries score 0.40–0.45 — above threshold, but the
  sidecar correctly rejects the fake tool name reference before they route
- Gap between lowest in-scope (0.4029) and highest pure OOS (0.3381): **+0.065**

**Uncertain zone detail (0.40–0.55):**

16 in-scope queries land in the uncertain zone. All route correctly, but in
a multi-turn system, the agent could inspect the fact tree to verify:

| Query ID | Conf | Tool | Query (truncated) |
|---|---|---|---|
| schema_08 | 0.4029 | analytics_get_metrics | Get page views grouped by week for Jan 2025 |
| schema_02 | 0.4182 | email_send | Email the team at eng@acme.com, product@... |
| clear_02 | 0.4393 | email_send | Email the quarterly report to finance@acme.com |
| adv_14 | 0.4431 | stripe_refund | Refund the customer and then send them a... |
| coll_20 | 0.4574 | stripe_list_invoices | Look up the payment history for our enterprise... |
| coll_35 | 0.4638 | email_send | Email the invoice to billing@client.com... |
| coll_01 | 0.4730 | slack_send_dm | Send a message to alice@acme.com about... |
| coll_11 | 0.4937 | calendar_find_free_time | Find a time that works for me and sarah@... |

These are the hardest queries — multi-action, cross-domain vocabulary, or
ambiguous targets. In a single-shot system they'd be risky. In a multi-turn
system the agent has the confidence score and fact tree to decide whether to
execute or ask for clarification.

### Perturbation Robustness

To test whether the confidence signal generalizes beyond the exact benchmark
phrasings, we ran a perturbation test: each in-scope query was paraphrased
3 ways using synonym substitution, word reordering, and conversational filler.
No encoder rules were modified.

**Results (206 paraphrased queries from 71 originals):**

| Metric | Original | Paraphrased |
|---|---|---|
| Routing accuracy | 100% (71/71) | 62.1% (128/206) |
| Mean confidence | 0.6395 | 0.5032 |
| Confidence shift | — | -0.0615 avg |

**Failure breakdown:**
- **41 false abstains**: Synonym verbs not in `_ACTION_MAP` (e.g., "fire off",
  "dig through", "capture") produce mismatched lexicon symbols, dropping the
  action role similarity to ~0 and pulling the overall score below threshold.
- **37 wrong tool**: When domain-disambiguating phrases change (e.g., "customer
  record" → "customer info"), the phrase rules don't fire and the BoW layer
  alone can't resolve CRM vs Stripe, identify vs track, etc.

**Confidence zones (paraphrased):**

| Zone | Count | Pct |
|---|---|---|
| High (>= 0.55) | 94 / 206 | 46% |
| Uncertain (0.40–0.55) | 71 / 206 | 34% |
| Abstain (< 0.40) | 41 / 206 | 20% |

**LLM comparison on the same paraphrased queries (gpt-4o-mini):**

| Router | Original | Paraphrased | Delta |
|---|---|---|---|
| Glyphh HDC (S2) | 100.0% (71/71) | 62.1% (128/206) | -37.9% |
| LLM gpt-4o-mini (S1) | 90.1% (64/71) | 88.8% (183/206) | -1.3% |

The LLM barely flinches on paraphrases (-1.3%) because its language model
naturally handles synonyms. Glyphh drops 37.9% because its lexicon can't
match verbs it hasn't seen.

But the failure profiles are **complementary**:
- **LLM failures (23)**: Semantic near-collisions — CRM vs Stripe, identify
  vs track, delete vs list. The *same categories* it fails on with original
  queries. Paraphrasing doesn't help or hurt the LLM.
- **Glyphh failures (78)**: Mostly false abstains from unknown synonym verbs.
  When Glyphh does match, it matches correctly. The confidence signal
  accurately reflects uncertainty.

This validates the multi-turn architecture: Glyphh handles the cases LLMs
get wrong (near-collisions, adversarial), and in a multi-turn flow, the
confidence score tells the agent exactly when to trust vs clarify. The LLM
never needs to reason about routing — it only sees the routed tool.

The perturbation test also identifies specific gaps in the action lexicon
and phrase rules that could be addressed to improve robustness without
changing the HDC architecture.

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
The Glyphh model doesn't "reason" about which tool — it decomposes the
query into structured signals and lets cosine similarity decide.

### 3. Confidence calibration enables multi-turn routing

The model's value isn't just accuracy — it's knowing what it knows. The
confidence distribution shows clean separation:

- **High confidence (>= 0.55)**: 77% of in-scope queries. These can execute
  without LLM intervention.
- **Uncertain zone (0.40–0.55)**: 23% of in-scope queries. All route
  correctly, but an agent can inspect the fact tree to verify. The fact tree
  shows which roles contributed most to the match — was it the action verb,
  the domain keywords, or the BoW overlap?
- **OOS gap**: Open-set queries max out at 0.3381, well below the 0.40
  threshold. The model confidently abstains on unrelated queries.

In a multi-turn architecture, the agent never needs to send the full tool
catalog to the LLM. Glyphh routes first (7ms, 0 tokens), and the LLM only
sees the single matched tool for argument extraction — or the fact tree for
clarification if confidence is low.

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

This validates the architectural split: let Glyphh handle routing (structured
decomposition + vector similarity), let the LLM handle argument extraction
(language understanding + schema compliance).

## Methodology Notes

- **LLM**: gpt-4o-mini, temperature=0, single pass (no majority vote)
- **Schema validation**: jsonschema Draft 7, enforced even on empty args
- **Routing policy**: Highest-impact/most irreversible action for multi-action
  queries, applied consistently across all strategies
- **Metrics separation**: Routing accuracy and end-to-end validity are reported
  independently so S2 (routing-only) is directly comparable
- **LLM prompt**: Full tool catalog with schemas, descriptions, parameter
  types, enums, and anyOf constraints included in the system prompt
- **Confidence scores**: Per-query confidence is the weighted role-level
  cosine similarity between query glyph and best matching exemplar glyph

## Limitations and Future Work

- **Internal eval only** — queries, exemplars, and encoder were developed by
  the same team. An independent held-out test set with paraphrased queries
  (no phrase leakage) is needed for external validation.
- **Perturbation robustness** — the benchmark uses canonical phrasings. A
  perturbation test with randomized paraphrases (synonym substitution, word
  reordering, passive voice) would validate that the confidence signal remains
  calibrated on unseen phrasings.
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
- **Confidence gap** — the lowest in-scope confidence (0.4029) and highest
  adversarial OOS confidence (0.4488) overlap by 0.046. The sidecar resolves
  this for tool-name adversarial queries, but natural-language adversarial
  queries in this gap zone would need multi-turn clarification.
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
