#!/bin/bash

set -euo pipefail

PROFILE="preprod"
if [[ "${1:-}" == "--profile" ]]; then PROFILE="${2:-preprod}"; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="agent-playbook-ai-api"
WORKSPACE_PATH="/Workspace/Users/contributor@example.com/$APP_NAME"

cd "$SCRIPT_DIR"

echo "── Uploading app files (profile: $PROFILE) ──"
databricks workspace mkdirs "$WORKSPACE_PATH" --profile "$PROFILE"
databricks workspace mkdirs "$WORKSPACE_PATH/wheels" --profile "$PROFILE"

FILES=(app.py app.yaml requirements.txt wheels/*.whl)
for f in "${FILES[@]}"; do
    echo "  ↑ $f"
    databricks workspace import "$WORKSPACE_PATH/$f" --file "$f" \
        --format AUTO --overwrite --profile "$PROFILE"
done

echo ""
echo "── Deploying app ──"
databricks apps deploy "$APP_NAME" \
    --source-code-path "$WORKSPACE_PATH" \
    --profile "$PROFILE"

echo ""
echo "✅ Done"
echo ""
echo "NOTE: run ingestion/ingest.py once on Databricks BEFORE first use to build:"
echo "  - \${AIF_TABLE:-ai_framework} chunks + vector index + summary index (from Confluence space PLAYBOOKIA)"
