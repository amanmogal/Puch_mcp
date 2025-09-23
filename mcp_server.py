"""
MCP Server for Puch AI Hiring Platform

The server includes tools for:
- Resume processing and serving
- Portfolio management from local files or GitHub
- Server metrics and health monitoring
- User feedback collection

Author: Aman Mogal
"""

import contextlib
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import TokenVerifier, AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import AnyUrl, AnyHttpUrl, Field, BaseModel
from pathlib import Path
import os
import time
import json
import asyncio
import markdownify
import readabilipy
from httpx import AsyncClient, HTTPError
from datetime import datetime, timezone
from typing import Annotated, Tuple

# Server configuration - Environment variables with fallback defaults
TOKEN = os.getenv("MCP_TOKEN")  
MY_NUMBER = os.getenv("MY_NUMBER", "918669427514")  

# Metrics tracking for monitoring server health and usage
SERVER_START_TS = time.time()
REQUEST_COUNT = 0
RECENT_ACTIONS = []
_METRICS_LOCK = asyncio.Lock()

async def _record_event(tool: str, status: str, duration_ms: float):
    """
    Record a tool execution event for metrics tracking.

    Args:
        tool: Name of the tool that was executed
        status: Execution status ('ok' or 'error')
        duration_ms: Execution time in milliseconds
    """
    async with _METRICS_LOCK:
        global REQUEST_COUNT
        REQUEST_COUNT += 1
        RECENT_ACTIONS.append({"tool": tool, "status": status, "duration_ms": round(duration_ms, 2)})
        if len(RECENT_ACTIONS) > 100:
            RECENT_ACTIONS.pop(0)

class RichToolDescription(BaseModel):
    """Enhanced tool description with usage guidance."""
    description: str
    use_when: str
    side_effects: str | None

class SimpleTokenVerifier(TokenVerifier):
    """Simple token-based authentication verifier."""

    def __init__(self, token: str):
        self.token = token

    async def verify_token(self, token: str):
        """Verify the provided token against the expected token."""
        print(f"Verifying token: {token}")
        if token == self.token:
            return AccessToken(token=token, client_id="dev", scopes=[])
        print(f"Invalid token: {token} != {self.token}")
        raise ValueError("Invalid token")

class DebugMiddleware(BaseHTTPMiddleware):
    """
    Debug middleware for logging requests and handling authentication.

    This middleware logs all HTTP requests and their headers, and performs
    Bearer token authentication with support for multiple header formats.
    """

    async def dispatch(self, request, call_next):
        """Process each HTTP request through the middleware."""
        print(f"Request: {request.method} {request.url}")
        print(f"Headers: {dict(request.headers)}")

        auth_header = request.headers.get('authorization', '')

        # Check for alternative token header formats if no standard auth header
        if not auth_header:
            for header_name in ['mcp-token', 'mcp_token', 'x-mcp-token', 'MCP_TOKEN', 'X-MCP-Token']:
                if header_name in request.headers:
                    token_value = request.headers[header_name]
                    # Create a mutable copy of headers to modify
                    new_headers = dict(request.headers)
                    new_headers['authorization'] = f"Bearer {token_value}"
                    # Update the request scope with the new headers
                    request.scope['headers'] = [(k.lower().encode(), v.encode()) for k, v in new_headers.items()]
                    auth_header = new_headers['authorization']
                    print(f"Mapped {header_name} to Authorization")
                    break

        # Validate the authorization header
        if not auth_header.startswith('Bearer '):
            print("No valid Authorization header")
            return JSONResponse({"error": "Authentication required"}, status_code=401)

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        if token != TOKEN:
            print(f"Invalid token: {token}")
            return JSONResponse({"error": "invalid_token", "error_description": "Authentication required"}, status_code=401)

        print(f"Auth ok for token: {token}")

        response = await call_next(request)
        print(f"Response: {response.status_code}")
        return response

async def debug_endpoint(request):
    """Debug endpoint that echoes request information."""
    return JSONResponse({
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "query_params": dict(request.query_params),
    })

class Fetch:

    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(cls, url: str, user_agent: str, force_raw: bool = False) -> Tuple[str, str]:
        """
        Fetch content from a URL.

        Args:
            url: The URL to fetch
            user_agent: User agent string for the request
            force_raw: If True, return raw content without HTML processing

        Returns:
            Tuple of (content, prefix) where prefix contains fetch metadata
        """
        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
                response.raise_for_status()
            except HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))
            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text
            content_type = response.headers.get("content-type", "")
            is_page_html = "text/html" in content_type.lower()

            if is_page_html and not force_raw:
                ret = readabilipy.simple_json_from_html_string(page_raw, use_readability=True)
                if not ret.get("content"):
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message="Page failed to be simplified from HTML"))
                if ret["content"] is not None:
                    content = markdownify.markdownify(ret["content"], heading_style="ATX")
                else:
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message="Page failed to be simplified from HTML (no content)"))
                prefix = f"Fetched {url} ({response.status_code})\n"
            else:
                content = page_raw
                prefix = f"Raw content from {url} ({response.status_code})\n"

            return content, prefix

# Initialize the MCP server
mcp = FastMCP("My MCP Server")

def text_to_markdown(text: str) -> str:
    """
    Convert plain text resume to Markdown format with basic formatting.

    Processes text line by line to identify headers, lists, and paragraphs,
    converting them to appropriate Markdown syntax.
    """
    lines = text.strip().split('\n')
    markdown_lines = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                in_list = False
            markdown_lines.append("")
            continue

        # Detect headers (ALL CAPS or resume section keywords)
        if line.isupper() or line.lower().startswith(("name:", "education:", "experience:", "skills:", "projects:")):
            markdown_lines.append(f"## {line}")
            in_list = False
        # Detect list items (starting with -, *, or numbers)
        elif line.startswith(('-', '*', '1.', '2.', '3.')):
            markdown_lines.append(f"- {line.lstrip('-*0123456789.').strip()}")
            in_list = True
        # Treat as regular paragraph
        else:
            if in_list:
                in_list = False
                markdown_lines.append("")
            markdown_lines.append(line)

    return "\n".join(markdown_lines).strip()

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown, converted from a text file.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """Return your resume as markdown text, converted from a text file."""
    try:
        resume_path = Path(__file__).parent / "resume.txt"
        if not resume_path.exists():
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="resume.txt not found in server directory"))
        
        text = resume_path.read_text(encoding="utf-8")
        if not text.strip():
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="resume.txt is empty"))
        
        # Convert plain text to Markdown
        markdown_resume = text_to_markdown(text)
        return markdown_resume
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to load or convert resume: {e!r}"))

@mcp.tool()
async def validate() -> str:
    """Return phone number for validation."""
    if not MY_NUMBER:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Phone number not configured"))
    return MY_NUMBER

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its raw content.",
    use_when="Use when you need the exact textual contents of a URL.",
    side_effects=None,
)

@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[int, Field(default=5000, description="Maximum number of characters to return.", gt=0, lt=1000000)] = 5000,
    start_index: Annotated[int, Field(default=0, description="Start output at this character index.", ge=0)] = 0,
    raw: Annotated[bool, Field(default=False, description="Return raw content without prefix")] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url_str:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))
    
    start_t = time.perf_counter()
    try:
        content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
        original_length = len(content)
        if start_index >= original_length:
            content = "<error>No more content available.</error>"
        else:
            truncated_content = content[start_index : start_index + max_length]
            if not truncated_content:
                content = "<error>No more content available.</error>"
            else:
                content = truncated_content
                if len(truncated_content) == max_length and original_length > start_index + max_length:
                    next_start = start_index + max_length
                    content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
        await _record_event("fetch", "ok", (time.perf_counter() - start_t) * 1000)
        return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]
    except McpError:
        await _record_event("fetch", "error", (time.perf_counter() - start_t) * 1000)
        raise
    except Exception as e:
        await _record_event("fetch", "error", (time.perf_counter() - start_t) * 1000)
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Fetch failed: {e!r}"))

PortfolioToolDescription = RichToolDescription(
    description="Serve portfolio details from a local markdown file or a GitHub repository README.",
    use_when="Use when the user asks for projects/portfolio or wants to view GitHub README content.",
    side_effects=None,
)

def _parse_github_repo_string(repo: str) -> Tuple[str, str]:
    """Accepts 'owner/name' or a full GitHub URL and returns (owner, name)."""
    s = repo.strip()
    if not s:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Empty github_repo provided"))
    if "github.com" in s:
        try:
            parts = s.split("github.com", 1)[1].strip("/")
            owner, name = parts.split("/", 2)[:2]
            return owner, name
        except Exception:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid GitHub URL format. Use owner/name or a standard GitHub URL."))
    if "/" not in s:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="github_repo must be 'owner/name' or a GitHub URL"))
    owner, name = s.split("/", 1)
    return owner, name

@mcp.tool(description=PortfolioToolDescription.model_dump_json())
async def portfolio(
    github_repo: Annotated[
        str | None,
        Field(
            default=None,
            description="GitHub repo as 'owner/name' or full GitHub URL. If omitted, falls back to local portfolio.md or env PORTFOLIO_GITHUB_REPO.",
        ),
    ] = None,
    branch: Annotated[
        str,
        Field(default="HEAD", description="Git branch or ref to read README from."),
    ] = "HEAD",
    readme_path: Annotated[
        str,
        Field(default="README.md", description="Path to README within the repo."),
    ] = "README.md",
    prefer_local: Annotated[
        bool,
        Field(default=True, description="Return local portfolio.md if present."),
    ] = True,
) -> str:
    """Return portfolio content."""
    try:
        root = Path(__file__).parent
        local_file = root / "portfolio.md"
        start_t = time.perf_counter()
        if prefer_local and local_file.exists():
            try:
                text = local_file.read_text(encoding="utf-8")
                if not text.strip():
                    raise McpError(ErrorData(code=INTERNAL_ERROR, message="portfolio.md is empty."))
                await _record_event("portfolio", "ok", (time.perf_counter() - start_t) * 1000)
                return text
            except McpError:
                await _record_event("portfolio", "error", (time.perf_counter() - start_t) * 1000)
                raise
        repo_str = github_repo or os.getenv("PORTFOLIO_GITHUB_REPO")
        if not repo_str:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="No portfolio.md and no GitHub repo provided. Add portfolio.md or pass github_repo (owner/name) or set PORTFOLIO_GITHUB_REPO."
                )
            )
        owner, name = _parse_github_repo_string(repo_str)
        raw_url = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{readme_path}"
        try:
            content, _prefix = await Fetch.fetch_url(raw_url, Fetch.USER_AGENT, force_raw=True)
            if not content.strip():
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="Fetched README is empty."))
            links_block = ("\n\n---\nLinks:\n- GitHub: https://github.com/amanmogal\n- Website: https://aman-mogal.vercel.app/\n")
            if "github.com/amanmogal" not in content or "aman-mogal.vercel.app" not in content:
                content = content.rstrip() + links_block
            await _record_event("portfolio", "ok", (time.perf_counter() - start_t) * 1000)
            return content
        except McpError:
            await _record_event("portfolio", "error", (time.perf_counter() - start_t) * 1000)
            raise
    except Exception as e:
        await _record_event("portfolio", "error", (time.perf_counter() - start_t) * 1000)
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to load portfolio: {e!r}"))

MetricsToolDescription = RichToolDescription(
    description="Return basic server metrics: uptime, request count, and recent actions.",
    use_when="Use to inspect server health and recent tool activity.",
    side_effects=None,
)

@mcp.tool(description=MetricsToolDescription.model_dump_json())
async def metrics() -> str:
    """Return metrics as a JSON string."""
    uptime_seconds = max(0.0, time.time() - SERVER_START_TS)
    async with _METRICS_LOCK:
        recent = list(RECENT_ACTIONS)
        count = REQUEST_COUNT
    payload = {
        "uptime_seconds": round(uptime_seconds, 2),
        "request_count": count,
        "recent_actions": recent,
    }
    return json.dumps(payload, ensure_ascii=False)

class FeedbackItem(BaseModel):
    """Data model for user feedback submissions."""
    name: str | None = None
    rating: Annotated[int, Field(ge=1, le=5, description="1-5 stars")]
    comments: str | None = None
    contact: str | None = None

@mcp.tool()
async def feedback(item: FeedbackItem) -> str:
    """Accept feedback and store to a JSONL file."""
    start = time.perf_counter()
    path = os.getenv("FEEDBACK_FILE", "feedback.jsonl")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "name": item.name,
        "rating": item.rating,
        "comments": item.comments,
        "contact": item.contact,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        await _record_event("feedback", "ok", (time.perf_counter() - start) * 1000)
        return "Thanks for your feedback."
    except Exception as e:
        await _record_event("feedback", "error", (time.perf_counter() - start) * 1000)
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to store feedback: {e!r}"))

class AuthMiddleware:
    """
    ASGI middleware for authentication and header mapping.

    Handles Bearer token authentication and maps various MCP client
    token header formats to the standard Authorization header.
    """

    def __init__(self, app):
        """Initialize the middleware with the wrapped ASGI app."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Process ASGI requests and perform authentication."""
        if scope['type'] == 'http':
            headers = dict((k.decode(), v.decode()) for k, v in scope['headers'])
            print(f"Request: {scope['method']} {scope['path']}")
            print(f"Headers: {headers}")

            auth_header = headers.get('authorization', '')
            if not auth_header:
                # Map alternative token headers to standard Authorization header
                for header_name in ['mcp-token', 'mcp_token', 'x-mcp-token', 'MCP_TOKEN', 'X-MCP-Token']:
                    if header_name in headers:
                        headers['authorization'] = f"Bearer {headers[header_name]}"
                        scope['headers'] = [(k.encode(), v.encode()) for k, v in headers.items()]
                        print(f"Mapped {header_name} to Authorization")
                        auth_header = headers['authorization']
                        break

            # Validate Bearer token
            if not auth_header.startswith('Bearer '):
                print("No valid Authorization header")
                await send({'type': 'http.response.start', 'status': 401, 'headers': [(b'content-type', b'application/json')]})
                await send({'type': 'http.response.body', 'body': b'{"error": "Authentication required"}'})
                return

            token = auth_header[7:]  # Extract token after 'Bearer '
            if token != TOKEN:
                print(f"Invalid token: {token}")
                await send({'type': 'http.response.start', 'status': 401, 'headers': [(b'content-type', b'application/json')]})
                await send({'type': 'http.response.body', 'body': b'{"error": "invalid_token"}'})
                return

            print(f"Auth ok for token: {token}")

        # Forward request to the wrapped app
        await self.app(scope, receive, send)

# Initialize the ASGI application with authentication middleware
app = AuthMiddleware(mcp.streamable_http_app())

# Deployment note: Clients connect to https://your-app.vercel.app/mcp

# Development server - only runs when script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))