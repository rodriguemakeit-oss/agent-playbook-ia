# Agent Playbook

A FastAPI service for document-grounded chat, streaming responses, and structured tool events.

## What it demonstrates

- An OpenAI-compatible chat completions endpoint
- Server-sent events (SSE) for streaming answers and tool activity
- A modular ingestion and retrieval layout
- Contract-focused API tests

## Repository map

- `app.py` — API application
- `ingestion/` — document ingestion components
- `tests/` — API and streaming contract checks

## Run locally

See [GUIDE.md](GUIDE.md) for setup. The service requires your own retrieval backend, credentials, and data sources. No hosted service or document corpus is included.

## Scope

This is a portfolio adaptation for local review. Configuration examples are placeholders; cloud integration and production readiness must be validated in your environment. See [VALIDATION.md](VALIDATION.md) for the checks performed and their limits.