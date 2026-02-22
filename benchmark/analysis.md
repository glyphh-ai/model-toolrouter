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
| adversarial | 15 | Multi-action, fake tool names, misleading phrasing |
| open_set | 15 | Out-of-scope queries that should return null (no tool) |
| schema_trap | 10 | Correct tool is obvious but args are tricky (cents, enums, anyOf) |

## Strategies

| # | Strategy | Description |
|---|---|---|
| S1 | LLM Only | LLM selects tool + generates args from full catalog |
| S2 | Glyphh Only | HDC encoder routes to tool, no args (routing-only baseline) |
| S3 | Glyphh Route + LLM Args | Glyphh selects tool, LLM generates args for that single tool |
| S4 | LLM + Glyphh Sidecar | LLM selects tool + args, Glyphh confirms or overrides routing |

## Results

### Routing Accuracy

How often each strategy picks the correct tool (or correctly abstains).

| Strategy | Tool Acc | In-Scope | Abstain | Latency (avg) | Tokens |
|---|---|---|---|---|---|
| S1: LLM Only | 91.6% | 88.7% | 100% | 1,820 ms | 229K |
| S2: Glyphh Only | 100% | 100% | 100% | 6 ms | 0 |
| S3: Glyphh Route + LLM Args | 100% | 100% | 100% | 658 ms | 22K |
| S4: LLM + Glyphh Sidecar | 98.9% | 100% | 95.8% | 1,655 ms | 232K |

### End-to-End Validity (correct tool + schema-valid args)

Only applies to strategies that produce arguments. Schema validation uses
jsonschema Draft 7.

| Strategy | E2E Acc | Schema Pass Rate |
|---|---|---|
| S1: LLM Only | 88.7% | 100% (69/69) |
| S3: Glyphh Route + LLM Args | 100% | 100% (71/71) |
| S4: LLM + Glyphh Sidecar | 100% | 100% (72/72) |

S2 is excluded — it produces no args by design.

## Key Findings

### 1. LLMs struggle with semantic near-collisions

S1 gets 8 routing decisions wrong. The failures are systematic, not random:

- **CRM vs Stripe**: "Find the customer record for bob@startup.io" → LLM picks
  `stripe_get_customer` instead of `crm_get_contact`. The word "customer"
  pulls it toward Stripe.
- **track vs identify**: "Log that user u_555 upgraded to pro" → LLM picks
  `analytics_track_event` instead of `analytics_identify_user`. Plan changes
  are trait updates, not events.
- **delete vs list**: "Remove the standup meeting" → LLM picks
  `calendar_list_events` (reasoning: "need to find it first"). Correct under
  Policy C: the highest-impact action is the deletion.
- **Multi-action routing**: Under Policy C (highest-impact action), the LLM
  sometimes picks the notification instead of the state change.

These are exactly the cases where structured intent decomposition
(action + target + domain) outperforms token-level pattern matching.

### 2. Glyphh routes perfectly with zero LLM dependency

S2 achieves 100% routing accuracy using only HDC cosine similarity on intent
vectors. No API calls, no tokens, sub-10ms latency. The encoder decomposes
each query into action, target, domain, and keyword signals, then matches
against 68 exemplars using weighted role similarity.

### 3. S3 is the cost-efficiency play

S3 matches S4's accuracy while using 90% fewer tokens and running 64% faster
than S1. The LLM never sees the full 38-tool catalog — it receives a single
tool definition and extracts args. This is the architecture for high-volume,
cost-sensitive deployments.

| Metric | S1 (LLM Only) | S3 (Glyphh + LLM) | Reduction |
|---|---|---|---|
| Tokens | 229K | 22K | 90% |
| Latency | 1,820 ms | 658 ms | 64% |
| Routing errors | 8 | 0 | 100% |

### 4. S4 sidecar catches every in-scope LLM mistake

On the 71 queries that map to real tools, S4 is 100%. Glyphh overrides the
LLM's wrong pick on every near-collision and adversarial case. The single
failure is an out-of-scope query (`oos_09`: "Summarize the last 5 support
tickets") where the LLM false-positived on `jira_search` and Glyphh correctly
abstained — but the sidecar's design trusts the LLM when Glyphh has no
opinion. This is a known, documented trade-off.

### 5. The LLM is good at args, bad at routing

Schema validity is 100% across all strategies. Once the LLM knows which tool
to call, it fills in fields correctly — types, enums, required fields, cents
conversion, anyOf constraints. The problem is purely in the routing decision
over a large catalog.

## Methodology Notes

- **LLM**: gpt-4o-mini, temperature=0, single pass (no majority vote)
- **Schema validation**: jsonschema Draft 7, enforced even on empty args
- **Routing policy**: Policy C — highest-impact/most irreversible action for
  multi-action queries, applied consistently across all strategies
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
- **S4 OOS trade-off** — the sidecar trusts the LLM when Glyphh abstains,
  which can propagate LLM false positives on out-of-scope queries.

## Version

- Benchmark: v2.2.0
- Tool catalog: 38 tools, 8 domains
- Query set: 95 queries, 5 categories
- Encoder: 10,000-dimensional HDC, 68 exemplars
- Routing policy: Policy C (highest-impact action)
