"""One isolated robot process; identical MCP surface for simulation and hardware."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import importlib
import json
import secrets
import sys
from pathlib import Path


async def serve(config: dict, port: int, token: str):
    import uvicorn
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.types import CallToolResult, ImageContent, TextContent, Tool
    from robot_rollout.robot_control import RobotControl
    from starlette.applications import Starlette
    from starlette.routing import Route

    module, factory = config["robot"]["driver"].split(":", 1)
    with contextlib.redirect_stdout(sys.stderr):
        driver = getattr(importlib.import_module(module), factory)(
            task=config["robot"]["task"], seed=config["robot"]["seed"]
        )
    control = RobotControl(driver, Path(config["directory"]), config["profile"], config["robot"]["command_budget"])
    # A model that stops without observing is a valid failed attempt, not a
    # missing-evidence infrastructure error. This frame is private until observe.
    control.snapshot()
    server = Server("slime-robotics")
    lock = asyncio.Lock()

    @server.list_tools()
    async def list_tools():
        return [Tool(**t) for t in control.tools.values()]

    @server.call_tool()
    async def call_tool(name, arguments):
        async with lock:
            try:
                # MuJoCo's render context stays on its creating thread.
                reply = control.dispatch(name, arguments)
                content = [TextContent(type="text", text=json.dumps(reply.data, allow_nan=False))]
                for frame in reply.images:
                    content.extend(
                        [
                            TextContent(
                                type="text", text=f"Camera {frame['camera']} ({frame['width']}x{frame['height']})"
                            ),
                            ImageContent(
                                type="image",
                                mimeType=frame["mime_type"],
                                data=base64.b64encode(frame["data"]).decode(),
                            ),
                        ]
                    )
                return CallToolResult(
                    content=content, structuredContent=reply.data, isError=reply.data.get("status") == "error"
                )
            except Exception as exc:
                control.log("tool_error", {"tool": name, "error": str(exc)})
                return CallToolResult(
                    content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")], isError=True
                )

    manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)

    class MCPEndpoint:
        async def __call__(self, scope, receive, send):
            headers = dict(scope["headers"])
            if not secrets.compare_digest(headers.get(b"authorization", b"").decode(), "Bearer " + token):
                from starlette.responses import Response

                await Response(status_code=401)(scope, receive, send)
                return
            await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        try:
            async with manager.run():
                yield
        finally:
            driver.close()

    app = Starlette(
        routes=[Route("/mcp", endpoint=MCPEndpoint(), methods=["GET", "POST", "DELETE"])], lifespan=lifespan
    )
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    asyncio.run(serve(config, args.port, config.pop("token")))


if __name__ == "__main__":
    main()
