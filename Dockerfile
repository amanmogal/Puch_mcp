# Use Python 3.11 to match requirements.txt specification
FROM python:3.11-slim

# Set environment variables for better Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Copy application code
COPY . .

# Switch to non-root user
USER app

# Expose port 3000 (Vercel standard for Docker deployments)
EXPOSE 3000

# Health check using Python (runtime check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import socket, os; port=int(os.getenv('PORT', 3000)); s=socket.socket(); s.settimeout(5); r=s.connect_ex(('localhost', port)); s.close(); exit(0 if r==0 else 1)"

# Run the MCP server with dynamic port
CMD ["sh", "-c", "uvicorn mcp_server:app --host 0.0.0.0 --port ${PORT:-3000}"]
