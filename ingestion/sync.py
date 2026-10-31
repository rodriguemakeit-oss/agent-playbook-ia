# Databricks notebook source

# MAGIC %pip install databricks-vectorsearch /Volumes/labanalysis_dev/default/smartchunking_test/smartchunking-0.5.7-py3-none-any.whl /Volumes/labanalysis_dev/default/smartchunking_test/smartagentic-0.2.3-py3-none-any.whl --force-reinstall --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

CATALOG = os.environ.get("AIF_CATALOG", "agent_playbook_ai")
SCHEMA = os.environ.get("AIF_SCHEMA", "default")
TABLE = os.environ.get("AIF_TABLE", "ai_framework")
VS_ENDPOINT = os.environ.get("AIF_VS_ENDPOINT", "smartchunking")

# 401802104 = "AI & Advanced Analytics Framework Knowledge Base" (the real content tree).
# NOT the space's auto-generated homepage (401615829, homepageId in its URL) — that page has
# no children, so pointing here used to ingest only the homepage boilerplate and nothing else.
ROOT_PAGE_ID = os.environ.get("AIF_ROOT_PAGE_ID", "401802104")
# NONE (no status decided yet) is treated as validated/kept, not excluded — excluding it
# turned out to be net negative: several still-NONE section pages (A. Concepts & Method,
# B. Data & Model Platforms, C. AI Assistants) sit above individually-VALIDATED children
# (A0, A1, A7, C7, Overview & GBU License-Attribution Policy), and the library's cascade
# exclusion wiped out those validated children along with their NONE-tagged parent section.
# Only WIP is excluded now. Known tradeoff: a few pages with a WIP/WORK IN PROGRESS visible
# label but a NONE tag (B. Data & Model Platforms, B1, C1, C2) get indexed anyway.
EXCLUDE_BODY_REGEX = os.environ.get("AIF_EXCLUDE_BODY_REGEX", r"AI-INDEX-STATUS\s*:\s*WIP")

CHUNK_STRATEGY = os.environ.get("AIF_CHUNK_STRATEGY", "agentic")
EMBEDDING_MODEL = os.environ.get("AIF_EMBEDDING_MODEL", "databricks-gte-large-en")
LLM_MODEL = os.environ.get("AIF_LLM_MODEL", "databricks-gpt-5-2")

CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "https://service.example.invalid")
CONFLUENCE_USER = os.environ.get("CONFLUENCE_USER", "contributor@example.com")
try:
    CONFLUENCE_TOKEN = dbutils.secrets.get("agent_playbook_scope", "confluence-token")
except Exception:
    CONFLUENCE_TOKEN = os.environ["CONFLUENCE_TOKEN"]

print(f"Syncing page {ROOT_PAGE_ID} (+ descendants) → {CATALOG}.{SCHEMA}.{TABLE}")

# COMMAND ----------

from smartchunking import ConfluenceParser, Chunker, VectorStore, Summarizer

parser = ConfluenceParser(base_url=CONFLUENCE_URL, username=CONFLUENCE_USER, api_token=CONFLUENCE_TOKEN)

chunker = Chunker(
    strategy=CHUNK_STRATEGY,
    embedding_model=EMBEDDING_MODEL,
    llm_model=LLM_MODEL,
)

store = VectorStore(
    catalog=CATALOG,
    schema=SCHEMA,
    table_name=TABLE,
    endpoint_name=VS_ENDPOINT,
    embedding_model=EMBEDDING_MODEL,
)

summarizer = Summarizer(
    catalog=CATALOG,
    schema=SCHEMA,
    table_name=TABLE,
    model=LLM_MODEL,
    embedding_model=EMBEDDING_MODEL,
)

result = parser.sync(
    store=store,
    chunker=chunker,
    root_page_id=ROOT_PAGE_ID,
    include_root=True,
    summarizer=summarizer,
    exclude_body_regex=EXCLUDE_BODY_REGEX,
)

print("Sync result:")
print(f"  added     : {len(result['added'])}  {result['added']}")
print(f"  updated   : {len(result['updated'])}  {result['updated']}")
print(f"  deleted   : {len(result['deleted'])}  {result['deleted']}")
print(f"  unchanged : {result['unchanged']}")
