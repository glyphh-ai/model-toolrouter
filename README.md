# DevOps Tool Router

Routes natural language DevOps requests to release step functions using HDC similarity matching on intent vectors.

## How It Works

You describe what you want to do in plain English — "create a release branch", "run the tests", "tag the release" — and the model routes your request to the correct tool function via cosine similarity on HDC-encoded intent vectors. No LLM in the loop. No hallucinations. Just math.

The exemplars in `data/exemplars.jsonl` define the tool catalog — each tool has multiple phrasings encoded as concepts. Your query is encoded the same way, and the closest match wins.

## Model Structure

```
model-toolrouter/
├── manifest.yaml          # model identity and metadata
├── config.yaml            # runtime config, roles, similarity settings
├── encoder.py             # EncoderConfig + encode_query + entry_to_record
├── build.py               # package model into .glyphh file
├── tests.py               # test runner entry point
├── data/
│   └── exemplars.jsonl    # tool exemplar definitions
├── tests/
│   ├── test-queries.json  # test query → expected tool mappings
│   ├── conftest.py        # shared fixtures
│   ├── test_encoding.py   # config validation, role encoding
│   ├── test_similarity.py # routing correctness
│   └── test_queries.py    # NL query attribute extraction
└── README.md
```

## Encoder Architecture

Two-layer HDC encoder tuned for intent routing:

| Layer | Weight | Segments | Purpose |
|-------|--------|----------|---------|
| intent | 0.7 | action (verb + object), scope (domain) | Primary routing signal |
| context | 0.3 | keywords | Disambiguation via keyword overlap |

### Roles

| Role | Type | Weight | Description |
|------|------|--------|-------------|
| verb | categorical | 1.0 | Action class: build, deploy, query, manage |
| object | text | 0.8 | Target: branch, tag, tests, sdk, runtime, etc. |
| domain | categorical | 1.0 | Context: release, test, build, docker |
| keywords | text | 1.0 | Filtered query terms for disambiguation |

## Tools

| Tool ID | Description |
|---------|-------------|
| create_branch | Create a release branch from main |
| run_tests | Run the test suite |
| merge_to_main | Merge the release branch into main |
| create_tag | Tag the release on main |
| check_workflow_status | Check CI/CD build status |
| cleanup_branch | Delete the release branch |
| release_sdk | Full SDK release flow |
| release_runtime | Full runtime release flow |
| release_platform | Full platform release flow |
| release_studio | Full studio release flow |
| rebuild_runtime | Rebuild runtime docker container |
| rebuild_studio | Rebuild studio docker container |
| rebuild_all | Rebuild all docker containers |
| restart_runtime | Restart runtime container |
| restart_all | Restart all containers |
| docker_logs | Show docker container logs |
| docker_status | Show docker container status |

## Testing

```bash
# Via CLI
glyphh model test ./model-toolrouter
glyphh model test ./model-toolrouter -v

# Or directly
cd model-toolrouter/
python tests.py
```

## Benchmark (coming soon)

A benchmark comparing Glyphh HDC routing vs. LLM-based tool selection on the same query set — measuring accuracy, latency, and cost.
