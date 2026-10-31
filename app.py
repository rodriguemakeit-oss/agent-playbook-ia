"""Agent Playbook AI API — OpenAI-compatible agent backend.

Exposes `POST /v1/chat/completions` (SSE, OpenAI Chat Completions chunk format) so the
playbook FastAPI backend can consume it as an agent (see AGENT-BACKEND-INTEGRATION.md).

The agent answers questions about the demo AI Playbook (Confluence space PLAYBOOKIA),
powered by smartagentic's AgentRAG, using ONLY the pre-built Databricks tables/index:
  - search()          → semantic vector search over the page chunks
  - read_summaries()  → hierarchical summaries (global questions)
  - sql_query()       → SQL over chunks / summaries tables

Streaming: the agent streams progress AND the final answer token-by-token. Tool activity is
surfaced as **structured** events on `choices[0].delta.agent_event` (type `tool_call` /
`tool_result`, paired by a shared `id`) — NOT as markdown in `delta.content`. `delta.content`
carries only the real answer tokens (plus lightweight thinking/reformulate/mode notes and the
sources/usage footer). This lets the front render tool calls distinctly (Copilot-style) and
know exactly when a tool starts and ends (see §5/§6 of the integration guide).

Event contract (additive; plain OpenAI clients ignore the unknown `agent_event` field):
  delta.agent_event = {
    "v": 1, "seq": <int>, "iteration": <int>,
    "type": "tool_call" | "tool_result",
    "id": "tc_<n>",              # START (tool_call) and END (tool_result) share this id
    "tool": "search",
    "args": {...},                # tool_call only
    "status": "ok" | "error",    # tool_result only
    "length": <int>,             # tool_result only
    "preview": "<=200 chars>"    # tool_result only
  }
A tool chunk carries `agent_event` and NO `content`; an answer chunk carries `content` and NO
`agent_event` → the front separates the two trivially.

Confidence: every answer is scored 0-100 by a cheap separate judge LLM and surfaced as a
final `agent_event` of type `confidence` = { score, unsure, reason, logged }. When the score
is below AIF_CONFIDENCE_THRESHOLD (or the judge flags `unsure`), the Q/A is logged to the
Databricks table AIF_UNSURE_TABLE and a short note is streamed to the user in `content`.
"""

import base64
import json
import logging
import os
import pathlib
import tempfile
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from smartagentic import Querier

CATALOG = os.environ.get("AIF_CATALOG", "agent_playbook_ai")
SCHEMA = os.environ.get("AIF_SCHEMA", "default")
TABLE = os.environ.get("AIF_TABLE", "ai_framework")
VS_ENDPOINT = os.environ.get("AIF_VS_ENDPOINT", "smartchunking")

INDEX_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}_index"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"
SUMMARY_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}_summaries"

DEFAULT_MODEL = os.environ.get("AIF_MODEL", "databricks-claude-sonnet-4-6")
ALLOWED_MODELS = {
    "databricks-claude-sonnet-4-6",
    "databricks-claude-opus-4-6",
    "databricks-gpt-5-2",
}

os.environ.setdefault("DATABRICKS_WAREHOUSE_ID", os.environ.get("DATABRICKS_WAREHOUSE_ID", "1000000000000000"))
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

AIF_JUDGE_MODEL = os.environ.get("AIF_JUDGE_MODEL", "databricks-gpt-5-4-nano")
AIF_CONFIDENCE_THRESHOLD = int(os.environ.get("AIF_CONFIDENCE_THRESHOLD", "50"))
AIF_UNSURE_TABLE = os.environ.get("AIF_UNSURE_TABLE", f"{CATALOG}.{SCHEMA}.agent_unsure_answers")

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "").strip()

MAX_ATTACHMENT_BYTES = int(os.environ.get("AIF_MAX_ATTACHMENT_BYTES", str(2 * 1024 * 1024)))
_SCRATCH_DIR = pathlib.Path(__file__).parent / "_scratch"
_SCRATCH_DIR.mkdir(exist_ok=True)

_log_buffer = deque(maxlen=500)


class _BufferHandler(logging.Handler):
    def emit(self, record):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        _log_buffer.append(f"[{ts}] {record.levelname} {record.getMessage()}")


_handler = _BufferHandler()
_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_handler)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("agent-playbook-ai-api")

# Querier caches whatever Databricks auth token it grabs at construction time and never
# refreshes it (smartagentic issue, not ours to fix here — its Vector Search client logs
# "Using a Personal Authentication Token (PAT)" and just keeps reusing that string). Since we
# cached the same instance forever, search() would start failing with "Invalid Token" once
# that token aged out (observed after anywhere from hours to several days of app uptime).
# Fix: give each cached Querier a TTL — past it, drop and recreate, which mints a fresh token
# the same way a full app redeploy does, but transparently and without any downtime.
_QUERIER_TTL_SECONDS = int(os.environ.get("AIF_QUERIER_TTL_SECONDS", str(45 * 60)))
_queriers: dict[str, tuple[Querier, float]] = {}


def get_querier(model: str) -> Querier:
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    cached = _queriers.get(model)
    if cached is not None:
        querier, created_at = cached
        if time.monotonic() - created_at < _QUERIER_TTL_SECONDS:
            return querier
        logger.info("Querier for model=%s past TTL (%ss) — recreating for a fresh token", model, _QUERIER_TTL_SECONDS)
    logger.info("Initializing Querier (model=%s)", model)
    querier = Querier(
        index_name=INDEX_NAME,
        endpoint_name=VS_ENDPOINT,
        chunks_table=CHUNKS_TABLE,
        summary_table=SUMMARY_TABLE,
        model=model,
    )
    _queriers[model] = (querier, time.monotonic())
    return querier


class Attachment(BaseModel):
    file_name: Optional[str] = Field(None, description="Original file name; its extension drives type detection.")
    mime_type: Optional[str] = Field(None, description="MIME type reported by the backend.")
    size_bytes: Optional[int] = Field(None, description="Original size in bytes.")
    truncated: bool = Field(False, description="True if content_b64 was truncated by the backend (2 MB cap).")
    content_b64: Optional[str] = Field(
        None, description="Base64-encoded file content. Decoded and parsed (UniversalParser) into the agent context."
    )
    message_id: Optional[str] = Field(None, description="Id of the message this attachment belongs to.")
    message_role: Optional[str] = Field(None, description="Role of that message.")


class Message(BaseModel):
    role: str = Field(..., description="Normalized to 'user' or 'assistant'.", examples=["user"])
    content: str = Field(..., description="Message text.", examples=["Quelles sont les grandes composantes du framework ?"])


class Metadata(BaseModel):
    session_id: Optional[str] = Field(None, description="Opaque session id (logging only).")
    user_id: Optional[str] = Field(None, description="Opaque user id (logging only).")
    agent_slug: Optional[str] = Field(None, description="Agent identifier from the backend.")
    attachments: list[Attachment] = Field(
        default_factory=list, description="Files uploaded by the user; decoded and injected into the agent context."
    )


class ChatRequest(BaseModel):
    model: str = Field(
        DEFAULT_MODEL,
        description="Requested model. Only whitelisted Databricks models are honored; any other value "
        "(e.g. 'mock-agent') falls back to the default.",
        examples=["databricks-claude-sonnet-4-6"],
    )
    stream: bool = Field(
        True,
        description="If true, responds with `text/event-stream` (OpenAI chunk format). If false, returns a single JSON completion.",
    )
    messages: list[Message] = Field(
        ...,
        description="Full conversation history in chronological order. The last user message is the question; earlier messages are context.",
    )
    metadata: Metadata = Field(default_factory=Metadata, description="Session metadata and attachments.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "databricks-claude-sonnet-4-6",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Quelles sont les grandes composantes du framework ?"}
                    ],
                    "metadata": {"session_id": "abc123", "attachments": []},
                }
            ]
        }
    }


bearer_scheme = HTTPBearer(
    auto_error=False,
    description="API key (contract mode API_KEY). Omit when the agent is behind the Databricks Apps OAuth proxy.",
)


def check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    if not AGENT_API_KEY:
        return
    if credentials is None or credentials.credentials != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _extract_attachment_text(att: dict) -> Optional[str]:
    b64 = att.get("content_b64")
    name = att.get("file_name") or "attachment"
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
    except Exception:
        logger.warning("Attachment %s: base64 decode failed", name)
        return None
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raw = raw[:MAX_ATTACHMENT_BYTES]

    suffix = pathlib.Path(name).suffix or ".txt"
    tmp = tempfile.NamedTemporaryFile(dir=_SCRATCH_DIR, suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        from smartchunking import UniversalParser

        docs = UniversalParser(source_path=tmp.name).parse()
        text = "\n\n".join(d.text for d in docs if getattr(d, "text", None))
        return text.strip() or None
    except Exception as e:
        logger.warning("Attachment %s: parse failed (%s)", name, e)
        try:
            return raw.decode("utf-8").strip() or None
        except Exception:
            return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _build_question(messages: list[dict], attachments: list[dict]) -> tuple[str, list[dict]]:
    norm = [
        {"role": ("assistant" if m.get("role") == "assistant" else "user"), "content": m.get("content", "")}
        for m in messages
        if m.get("content")
    ]
    if not norm:
        return "", []
    question = norm[-1]["content"]
    history = norm[:-1]

    if attachments:
        blocks = []
        for att in attachments:
            text = _extract_attachment_text(att)
            name = att.get("file_name", "attachment")
            if text:
                note = " (contenu tronqué)" if att.get("truncated") else ""
                blocks.append(f"--- Pièce jointe : {name}{note} ---\n{text}")
        if blocks:
            question = (
                "Contexte fourni par l'utilisateur via des pièces jointes :\n\n"
                + "\n\n".join(blocks)
                + "\n\n--- Question ---\n"
                + question
            )
    return question, history


_JUDGE_SYSTEM_PROMPT = """You evaluate whether an AI assistant ACTUALLY ANSWERED the user's question,
using the retrieved context it relied on. You are NOT judging how honest, polite or well-written
the reply is — only whether the question got a usable answer.

Return ONLY a JSON object, no prose, no code fences:
{"score": <integer 0-100>, "unsure": <true|false>, "reason": "<one short sentence>"}

"score" = how fully and reliably the QUESTION was answered:
- 90-100: directly and completely answered, grounded in the context.
- 70-89 : the core question is answered, only minor gaps.
- 40-69 : partial answer, notable gaps or significant hedging.
- 1-39  : does not really answer (vague, evasive, mostly caveats).
- 0     : explicitly does NOT answer — says it does not know, cannot find it, that the
          information is not in the corpus/documentation, or redirects the user elsewhere.

"unsure" MUST be true whenever the assistant did NOT actually answer the question: it says it
does not know, cannot find the information, the info is not in the corpus, asks the user to look
elsewhere, or only gives disclaimers / generic guidance instead of the answer. A polite,
honest or well-reasoned refusal is STILL {"unsure": true} with a LOW score, because the
question was not answered.
Set "unsure" to false ONLY when a usable, on-point answer to the question is actually provided.
"""

_judge_llm = None
_unsure_table_ready = False


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        from smartchunking.backends.llm import DatabricksLLM
        _judge_llm = DatabricksLLM(model=AIF_JUDGE_MODEL)
    return _judge_llm


def _parse_judge(raw: str) -> Optional[dict]:
    import re
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        score = max(0, min(100, int(round(float(data["score"])))))
        return {
            "score": score,
            "unsure": bool(data.get("unsure", False)),
            "reason": str(data.get("reason", "") or "")[:500],
        }
    except (KeyError, ValueError, TypeError):
        return None


def _judge_confidence(question: str, answer: str, context: str, tools: list) -> Optional[dict]:
    if not answer.strip():
        return None
    user = (
        f"QUESTION:\n{question[:4000]}\n\n"
        f"RETRIEVED CONTEXT (tool outputs):\n{(context or '(none)')[:6000]}\n\n"
        f"TOOLS USED: {', '.join(tools) if tools else '(none)'}\n\n"
        f"ASSISTANT ANSWER:\n{answer[:6000]}\n\n"
        "Evaluate and return the JSON object."
    )
    raw = _get_judge_llm().chat_simple(_JUDGE_SYSTEM_PROMPT, user, max_tokens=300, temperature=0.0)
    return _parse_judge(raw)


def _ensure_unsure_table() -> None:
    global _unsure_table_ready
    if _unsure_table_ready:
        return
    from databricks.sdk import WorkspaceClient
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {AIF_UNSURE_TABLE} ("
        "id STRING, ts TIMESTAMP, session_id STRING, model STRING, "
        "question STRING, answer STRING, confidence_score INT, unsure BOOLEAN, "
        "reason STRING, tool_calls STRING, iteration_count INT)"
    )
    res = WorkspaceClient().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=ddl, wait_timeout="30s"
    )
    if res.status.state.value != "SUCCEEDED":
        raise RuntimeError(getattr(res.status.error, "message", str(res.status)))
    _unsure_table_ready = True


def _log_unsure(row: dict) -> bool:
    try:
        _ensure_unsure_table()
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import StatementParameterListItem as P
        stmt = (
            f"INSERT INTO {AIF_UNSURE_TABLE} "
            "(id, ts, session_id, model, question, answer, confidence_score, unsure, reason, "
            "tool_calls, iteration_count) VALUES "
            "(:id, current_timestamp(), :session_id, :model, :question, :answer, :score, "
            ":unsure, :reason, :tool_calls, :iters)"
        )
        params = [
            P(name="id", value=row["id"]),
            P(name="session_id", value=row.get("session_id")),
            P(name="model", value=row.get("model")),
            P(name="question", value=row.get("question", "")[:20000]),
            P(name="answer", value=row.get("answer", "")[:20000]),
            P(name="score", value=str(row["score"]), type="INT"),
            P(name="unsure", value=str(bool(row["unsure"])).lower(), type="BOOLEAN"),
            P(name="reason", value=row.get("reason", "")[:500]),
            P(name="tool_calls", value=row.get("tool_calls", "")[:20000]),
            P(name="iters", value=str(row.get("iters", 0)), type="INT"),
        ]
        res = WorkspaceClient().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID, statement=stmt, parameters=params, wait_timeout="30s"
        )
        ok = res.status.state.value == "SUCCEEDED"
        if not ok:
            logger.warning("Unsure logging not SUCCEEDED: %s", res.status)
        return ok
    except Exception as e:
        logger.warning("Unsure logging failed: %s", e)
        return False


def _chunk(cid: str, model: str, content: Optional[str] = None,
           role: Optional[str] = None, finish_reason: Optional[str] = None,
           agent_event: Optional[dict] = None) -> str:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if agent_event is not None:
        delta["agent_event"] = agent_event
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_stream(question: str, history: list[dict], model: str, session_id: Optional[str] = None):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    querier = get_querier(model)
    logger.info("Query (model=%s): %s", model, question[:200])

    yield _chunk(cid, model, role="assistant", content="")

    answer_started = False
    seq = 0
    iteration = 0
    tool_seq = 0
    pending_ids: deque[str] = deque()
    try:
        for event in querier.ask_stream(question, mode="agent", history=history or None):
            etype = event.get("type")

            if etype == "reformulate":
                yield _chunk(cid, model, content=f"> 🔄 _Reformulation : {event['question']}_\n")
            elif etype == "mode":
                yield _chunk(cid, model, content=f"> 🧭 _Mode : {event['mode']}_\n")
            elif etype == "thinking":
                note = event.get("content", "") or ""
                if note.strip().lower().startswith("iteration"):
                    iteration += 1
                yield _chunk(cid, model, content=f"> 💭 _{note}_\n")
            elif etype == "tool_call":
                tool_seq += 1
                seq += 1
                tc_id = f"tc_{tool_seq}"
                pending_ids.append(tc_id)
                yield _chunk(cid, model, agent_event={
                    "v": 1, "seq": seq, "iteration": iteration,
                    "type": "tool_call", "id": tc_id,
                    "tool": event["tool"], "args": event.get("args", {}),
                })
            elif etype == "tool_result":
                seq += 1
                tc_id = pending_ids.popleft() if pending_ids else f"tc_{tool_seq}"
                preview = (event.get("preview") or "").replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                status = "error" if preview.lstrip().lower().startswith(
                    ("error", "erreur", "exception", "traceback", "❌")) else "ok"
                yield _chunk(cid, model, agent_event={
                    "v": 1, "seq": seq, "iteration": iteration,
                    "type": "tool_result", "id": tc_id,
                    "tool": event["tool"], "status": status,
                    "length": event.get("length", 0), "preview": preview,
                })
            elif etype == "answer_delta":
                answer_started = True
                yield _chunk(cid, model, content=event.get("content", ""))
            elif etype == "answer":
                response = event["response"]
                if not answer_started and getattr(response, "answer", ""):
                    yield _chunk(cid, model, content=response.answer)
                footer = ""
                sources = getattr(response, "sources", None) or []
                if sources:
                    footer += "\n\n---\n📄 **Sources :**\n"
                    for s in sources:
                        meta = getattr(s.chunk, "metadata", None) or {}
                        title = meta.get("title", "")
                        footer += f"- [{s.score:.3f}] {title} (`{s.chunk.document_id}`)\n"
                if getattr(response, "total_tokens", 0):
                    footer += (
                        f"\n\n_🪙 ~{response.total_tokens:,} tokens "
                        f"({response.prompt_tokens:,} in / {response.completion_tokens:,} out)_"
                    )
                if footer:
                    yield _chunk(cid, model, content=footer)

                tool_calls = getattr(response, "tool_calls", None) or []
                judge = None
                try:
                    judge = _judge_confidence(
                        question,
                        getattr(response, "answer", "") or "",
                        getattr(response, "prompt", "") or "",
                        [tc.get("tool") for tc in tool_calls if tc.get("tool")],
                    )
                except Exception as e:
                    logger.warning("Confidence judge failed: %s", e)
                if judge is not None:
                    seq += 1
                    unsure_final = judge["unsure"] or judge["score"] < AIF_CONFIDENCE_THRESHOLD
                    logged = False
                    if unsure_final:
                        logged = _log_unsure({
                            "id": uuid.uuid4().hex,
                            "session_id": session_id,
                            "model": model,
                            "question": question,
                            "answer": getattr(response, "answer", "") or "",
                            "score": judge["score"],
                            "unsure": unsure_final,
                            "reason": judge["reason"],
                            "tool_calls": json.dumps(tool_calls, ensure_ascii=False),
                            "iters": iteration,
                        })
                        note = (
                            "Question enregistrée pour amélioration._"
                            if logged else
                            "(échec de journalisation)._"
                        )
                        yield _chunk(
                            cid, model,
                            content=f"\n\n> ⚠️ _Confiance faible ({judge['score']}/100) — réponse incertaine. {note}\n",
                        )
                    yield _chunk(cid, model, agent_event={
                        "v": 1, "seq": seq, "iteration": iteration,
                        "type": "confidence", "score": judge["score"],
                        "unsure": unsure_final, "reason": judge["reason"], "logged": logged,
                    })
    except Exception as e:
        logger.error("Agent error: %s\n%s", e, traceback.format_exc())
        yield _chunk(cid, model, content=f"\n\n❌ _Erreur interne de l'agent : {e}_")

    yield _chunk(cid, model, finish_reason="stop")
    yield "data: [DONE]\n\n"


API_DESCRIPTION = """
Agent **OpenAI-compatible** qui répond aux questions sur le **demo AI Playbook**
(espace Confluence PLAYBOOKIA), propulsé par un RAG agentique s'appuyant uniquement sur
les tables/index Databricks pré-construits (recherche vectorielle, résumés hiérarchiques, SQL).

### Streaming
`POST /v1/chat/completions` renvoie un flux **`text/event-stream`** au format
*OpenAI Chat Completions chunk*, terminé par `data: [DONE]`.

Les **appels d'outils** sont émis comme événements **structurés** sur
`choices[0].delta.agent_event` (et **non** en markdown dans `delta.content`) :

- `type: "tool_call"` — **début** d'un outil : `{ id, tool, args, seq, iteration }`.
- `type: "tool_result"` — **fin** du même outil (même `id`) : `{ id, tool, status, length, preview }`.

Un chunk d'outil porte `agent_event` **sans** `content` ; un chunk de réponse porte
`content` **sans** `agent_event`. `delta.content` ne contient donc que la **vraie réponse**
(plus de brèves notes _thinking/reformulation/mode_ et le footer sources/coût). Les clients
OpenAI basiques ignorent le champ `agent_event`.

> **Breaking** : la trace d'outils n'est plus dans `delta.content`. Les clients qui
> l'affichaient via `content` doivent lire `delta.agent_event`.

### Taux de confiance
Chaque réponse est notée **0-100** par un modèle **juge** séparé (cheap) et exposée en fin de
flux via un `agent_event` de type `confidence` = `{ score, unsure, reason, logged }`. Si le
score est sous le seuil (`AIF_CONFIDENCE_THRESHOLD`, défaut 50) ou si `unsure` est vrai, la
question/réponse est journalisée dans la table Databricks `AIF_UNSURE_TABLE` et une note est
ajoutée à la réponse (`content`) pour informer l'utilisateur.

### Authentification
Si `AGENT_API_KEY` est configurée côté agent : envoyer `Authorization: Bearer <clé>`.
Derrière le proxy OAuth des Databricks Apps, l'authentification est gérée par la plateforme
(mode `OAUTH_CLIENT_CREDENTIALS`) et `AGENT_API_KEY` reste vide.

Voir `AGENT-BACKEND-INTEGRATION.md` pour le contrat complet.
"""

TAGS_METADATA = [
    {"name": "chat", "description": "Endpoint de complétion de chat (SSE, OpenAI-compatible)."},
    {"name": "ops", "description": "Santé et exploitation."},
]

from mcp.server.fastmcp import FastMCP, Context

mcp_server = FastMCP("agent-playbook", stateless_http=True, streamable_http_path="/")


@mcp_server.tool()
async def ask_playbook(question: str, ctx: Context) -> str:
    """Answer a question about the portfolio demo "AI Playbook"
    (Confluence space PLAYBOOKIA): reference architecture, Explore/Develop/Scale lifecycle,
    model recommendations, guardrails, governance. Runs the agentic RAG and returns the answer.

    Progress (iterations, tool calls, confidence) is streamed as MCP log notifications while the
    agent works; the final answer text is returned as a single result (MCP has no token streaming).
    """
    import anyio

    q, _ = _build_question([{"role": "user", "content": question}], [])
    if not q:
        return "No question provided."

    def _run() -> str:
        parts: list[str] = []
        querier = get_querier(DEFAULT_MODEL)
        for event in querier.ask_stream(q, mode="agent"):
            etype = event.get("type")
            if etype == "thinking":
                anyio.from_thread.run(ctx.info, f"💭 {event.get('content', '')}")
            elif etype == "tool_call":
                anyio.from_thread.run(ctx.info, f"🔧 {event['tool']}({json.dumps(event.get('args', {}), ensure_ascii=False)})")
            elif etype == "tool_result":
                anyio.from_thread.run(ctx.info, f"📄 {event['tool']} → {event.get('length', 0)} chars")
            elif etype == "answer_delta":
                parts.append(event.get("content", ""))
            elif etype == "answer":
                resp = event["response"]
                if not parts and getattr(resp, "answer", ""):
                    parts.append(resp.answer)
        return "".join(parts) or "(empty answer)"

    try:
        return await anyio.to_thread.run_sync(_run)
    except Exception as e:
        logger.error("MCP ask_playbook error: %s", e)
        return f"Agent error: {e}"


_mcp_app = mcp_server.streamable_http_app()


@asynccontextmanager
async def _lifespan(_app):
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(
    title="Agent Playbook AI API",
    version="1.1.0",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    contact={"name": "demo Data Squad"},
    lifespan=_lifespan,
)


@app.get("/health", tags=["ops"], summary="Healthcheck", response_description="Toujours {\"status\": \"ok\"} si le serveur tourne.")
def health():
    return {"status": "ok"}


@app.post(
    "/v1/chat/completions",
    tags=["chat"],
    dependencies=[Depends(check_auth)],
    summary="Chat completion (SSE, OpenAI-compatible)",
    response_description=(
        "Flux `text/event-stream` de chunks OpenAI. Les appels d'outils sont émis sur "
        "`choices[0].delta.agent_event` (`tool_call`/`tool_result` appariés par `id`), la "
        "réponse token par token sur `delta.content`, terminé par `data: [DONE]`. "
        "Si `stream=false`, un unique objet JSON `chat.completion`."
    ),
)
async def chat_completions(req: ChatRequest):
    attachments = [a.model_dump() for a in req.metadata.attachments]
    messages = [m.model_dump() for m in req.messages]
    question, history = _build_question(messages, attachments)

    if not question:
        raise HTTPException(status_code=400, detail="No user message provided")

    model = req.model if req.model in ALLOWED_MODELS else DEFAULT_MODEL

    if not req.stream:
        querier = get_querier(model)
        answer = ""
        try:
            resp = querier.ask(question, mode="agent", history=history or None)
            answer = resp.answer
        except Exception as e:
            logger.error("Agent error (non-stream): %s", e)
            answer = f"Erreur interne de l'agent : {e}"
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                ],
            }
        )

    return StreamingResponse(
        _sse_stream(question, history, model, session_id=req.metadata.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/logs", tags=["ops"], response_class=PlainTextResponse, summary="Logs récents (buffer mémoire, 500 lignes)")
def get_logs():
    return "\n".join(_log_buffer)


app.mount("/mcp", _mcp_app)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=port)
