"""Generate docs/openapi.json from the FastAPI app.

The OpenAPI schema only needs the route + model definitions, not the real agent
backends. We inject lightweight stubs for the heavy libraries so this script runs
anywhere with just `fastapi` installed (no Databricks connection required).

Usage:
    python export_openapi.py         # writes docs/openapi.json
"""

import json
import pathlib
import sys
import types


def _install_stubs() -> None:
    """Stub smartagentic / smartchunking so `import app` succeeds without the wheels."""
    if "smartagentic" not in sys.modules:
        sa = types.ModuleType("smartagentic")
        sa.Querier = type("Querier", (), {"__init__": lambda self, **kw: None})
        sa.ConfluenceClient = type("ConfluenceClient", (), {"__init__": lambda self, **kw: None})
        sys.modules["smartagentic"] = sa
    if "smartchunking" not in sys.modules:
        sc = types.ModuleType("smartchunking")
        sc.UniversalParser = type("UniversalParser", (), {"__init__": lambda self, **kw: None})
        sys.modules["smartchunking"] = sc
    if "mcp.server.fastmcp" not in sys.modules:
        # Schema export never starts the MCP transport or calls agent tools.
        class SchemaOnlyMCP:
            def __init__(self, *args, **kwargs):
                pass

            def tool(self):
                return lambda function: function

            def streamable_http_app(self):
                async def transport(scope, receive, send):
                    raise RuntimeError("MCP transport is unavailable in schema-only mode")
                return transport

        mcp = types.ModuleType("mcp.server.fastmcp")
        mcp.FastMCP = SchemaOnlyMCP
        mcp.Context = type("Context", (), {})
        sys.modules["mcp.server.fastmcp"] = mcp


def main() -> None:
    _install_stubs()
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import app as app_module  # noqa: E402  (after stubs)

    schema = app_module.app.openapi()
    out = pathlib.Path(__file__).parent / "docs" / "openapi.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(schema.get('paths', {}))} paths, OpenAPI {schema.get('openapi')})")


if __name__ == "__main__":
    main()
