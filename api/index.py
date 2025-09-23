import os
import sys
from pathlib import Path

# Add the project root to Python path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the ASGI app from mcp_server.py
from mcp_server import app

# Vercel will automatically detect and use this ASGI app