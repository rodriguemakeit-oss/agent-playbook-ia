#!/bin/bash

set -euo pipefail

PROFILE="preprod"
if [[ "${1:-}" == "--profile" ]]; then PROFILE="${2:-preprod}"; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="/Workspace/Users/contributor@example.com/agent-playbook-ai-api-sync"
JOB_NAME="agent-playbook-ai-api-sync-daily"
SECRET_SCOPE="agent_playbook_scope"
ENV_FILE="$SCRIPT_DIR/../.env"

cd "$SCRIPT_DIR"

echo "── Syncing Confluence token → secret scope $SECRET_SCOPE (profile: $PROFILE) ──"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi
if [[ -z "${CONFLUENCE_TOKEN:-}" ]]; then
    echo "  ✗ CONFLUENCE_TOKEN not set. Create $ENV_FILE (see .env.example)." >&2
    exit 1
fi
databricks secrets create-scope "$SECRET_SCOPE" --profile "$PROFILE" 2>/dev/null || true
TOKFILE="$(mktemp)"; printf '%s' "$CONFLUENCE_TOKEN" | tr -d '\n\r' > "$TOKFILE"
databricks secrets put-secret "$SECRET_SCOPE" confluence-token \
    --string-value "$(cat "$TOKFILE")" --profile "$PROFILE"
rm -f "$TOKFILE"
echo "  ✅ $SECRET_SCOPE/confluence-token updated from .env"

echo ""
echo "── Importing sync notebook (profile: $PROFILE) ──"
databricks workspace mkdirs "$WORKSPACE_DIR" --profile "$PROFILE"
databricks workspace import "$WORKSPACE_DIR/sync" --file sync.py \
    --language PYTHON --format SOURCE --overwrite --profile "$PROFILE"

echo ""
echo "── Creating / resetting daily job ──"
EXISTING_ID="$(databricks jobs list --name "$JOB_NAME" --output json --profile "$PROFILE" \
    | jq -r '.[0].job_id // empty')"

if [[ -n "$EXISTING_ID" ]]; then
    echo "  Job exists (id=$EXISTING_ID) → reset"
    databricks jobs reset "$EXISTING_ID" \
        --json "{\"job_id\": $EXISTING_ID, \"new_settings\": $(cat _job_sync_daily.json)}" \
        --profile "$PROFILE"
    echo "  ✅ Updated job $EXISTING_ID"
else
    echo "  Creating new job"
    databricks jobs create --json @_job_sync_daily.json --profile "$PROFILE"
    echo "  ✅ Created"
fi

echo ""
echo "Test a one-shot run now:"
echo "  databricks jobs submit --json @_run_sync.json --no-wait --profile $PROFILE"
