# Puch MCP Server

A Model Context Protocol (MCP) server deployed on Vercel, providing various tools for AI assistants.

## Endpoints

- **MCP Streamable HTTP**: `https://puch-mcp-alky-2hvxtj7d2-amans-projects-528db5b2.vercel.app/mcp`
- **Debug Endpoint** (no auth): `https://puch-mcp-alky-2hvxtj7d2-amans-projects-528db5b2.vercel.app/debug`
- **Server Logs** (no auth): `https://puch-mcp-alky-2hvxtj7d2-amans-projects-528db5b2.vercel.app/server-logs`

## Authentication

All requests to `/mcp` require Bearer token authentication:
- Header: `Authorization: Bearer fa1eb43415fa`

## Tools Available

- `resume`: Returns resume in markdown format
- `validate`: Returns phone number for validation
- `fetch`: Fetches URL content
- `portfolio`: Returns portfolio information
- `metrics`: Returns server metrics
- `feedback`: Accepts user feedback

## Testing with MCP Inspector

1. Start MCP Inspector: `npx @modelcontextprotocol/inspector`
2. Open `http://localhost:6274` in your browser
3. Set Endpoint: `https://puch-mcp-alky-2hvxtj7d2-amans-projects-528db5b2.vercel.app/mcp`
4. Set Token: `fa1eb43415fa`
5. Select a tool and run tests

## Testing with Postman

- Method: POST
- URL: `https://puch-mcp-alky-2hvxtj7d2-amans-projects-528db5b2.vercel.app/mcp`
- Headers:
  - `Authorization: Bearer fa1eb43415fa`
  - `Content-Type: application/json`
- Body (example for resume tool):
  ```json
  {
    "tool": "resume",
    "input": {}
  }
  ```

## Deployment

Deployed on Vercel with automatic redeployment on GitHub pushes to main branch.

## Local Development

```bash
pip install -r requirements.txt
python mcp_server.py
```

Server runs on `http://localhost:8000` with MCP at `/mcp`.
