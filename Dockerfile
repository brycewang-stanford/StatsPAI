# Runs the StatsPAI MCP server over stdio, installed from PyPI.
# Used by MCP directories (Glama) that start the server to list its tools.
#   docker build -t statspai-mcp .
#   docker run -i --rm -v "$PWD:/data" -e STATSPAI_MCP_DATA_ROOTS=/data statspai-mcp
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN pip install --no-cache-dir statspai

WORKDIR /data
ENTRYPOINT ["statspai-mcp"]
