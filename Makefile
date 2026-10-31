# Agent Playbook AI API — dev / docs helpers

.PHONY: help openapi serve-docs run

help:
	@echo "Targets:"
	@echo "  openapi     Regenerate docs/openapi.json from app.py (single source of truth)"
	@echo "  serve-docs  Serve the static ReDoc site (docs/) locally on :8080"
	@echo "  run         Run the API locally (uvicorn) on :8000"

# Regenerate the OpenAPI spec that powers BOTH the in-app docs (/docs, /redoc)
# and the static GitHub Pages site (docs/index.html). Run after changing the API.
openapi:
	python3 export_openapi.py

# Preview the static reference site locally (same rendering as GitHub Pages).
serve-docs:
	@echo "ReDoc → http://localhost:8080"
	cd docs && python3 -m http.server 8080

run:
	python3 app.py
