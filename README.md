# MCP Server for Puch

This is a Model Context Protocol (MCP) server implementation for Puch, p
## Features

- **Resume Tool**: Serves resume content converted from a text file to Markdow
- **Portfolio Tool**: Serves portfolio from local file or GitHub repository
- **Metrics Tool**: Provides server health and usage statistics
- **Feedback Tool**: Collects user feedback and stores in JSONL format


## Running Locally

1. Install dependencies:
   ```bash
   uv install
   ```

2. Run the server:
   ```bash
   uv run python mcp_server.py
   ```

The server will start on `http://localhost:8000`.

## Connecting with Inspector

1. Open the MCP Inspector
2. Set the server URL to: `https://your-app.vercel.app/mcp` (or `http://localhost:8000/mcp` for local)
3. Set the transport type to: `streamable-http`
4. Add custom header: `Authorization: Bearer <your_token>`
5. Click "Connect"

## Configuration

Environment variables:
- `MCP_TOKEN`: Authentication token (default: `fa1eb43415fa`)
- `MY_NUMBER`: Phone number for validation (default: `918669427514`)
- `PORTFOLIO_GITHUB_REPO`: GitHub repo for portfolio fallback (optional)
- `FEEDBACK_FILE`: Path for feedback storage (default: `feedback.jsonl`)

## Files

- `mcp_server.py`: Main server implementation
- `api/index.py`: Vercel serverless function entry point
- `vercel.json`: Vercel deployment configuration
- `resume.txt`: Resume content (create this file)
- `portfolio.md`: Portfolio content (optional)
- `requirements.txt`: Python dependencies

## API Endpoints

- `POST /mcp`: MCP streamable HTTP endpoint (authenticated)
- `GET /mcp`: SSE endpoint for receiving messages (authenticated)

## Debugging

The server logs all requests and authentication status to the console. Check the Vercel function logs in the dashboard for troubleshooting connection issues.

## Troubleshooting

### Vercel Deployment Issues

1. **Function Timeout**: Vercel has a 10-second timeout for free tier. Some MCP operations might be slow.
2. **Cold Starts**: First request after inactivity may be slower.
3. **Environment Variables**: Make sure they're set in Vercel dashboard, not locally.

### Connection Issues

1. **Wrong URL**: Use `https://your-app.vercel.app/mcp` (note the `/mcp` path)
2. **Missing Token**: Ensure `Authorization: Bearer <token>` header is set
3. **CORS**: Vercel handles CORS automatically

### Local Development

For local testing, use the Inspector with `http://localhost:8000/mcp` and transport type `streamable-http`.
