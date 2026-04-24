"""
Factory for creating API clients.
"""

import os
from typing import Optional

from src.api.client import AnalysisClient
from src.api.replicate_client import ReplicateClient
from src.api.server_client import ServerClient


def create_client(
    source: Optional[str] = None,
    replicate_token: Optional[str] = None,
    server_url: Optional[str] = None,
    server_api_key: Optional[str] = None
) -> AnalysisClient:
    """Create appropriate API client based on configuration.

    Args:
        source: "replicate" or "server". If None, reads from env.
        replicate_token: Replicate API token.
        server_url: Server URL.
        server_api_key: Server API key.

    Returns:
        Configured AnalysisClient.
    """
    source = source or os.environ.get("ANALYSIS_SOURCE", "replicate")

    if source == "replicate":
        return ReplicateClient(api_token=replicate_token)
    elif source == "server":
        if not server_url:
            raise ValueError("Server URL required when using server source")
        return ServerClient(server_url=server_url, api_key=server_api_key)
    else:
        raise ValueError(f"Unknown analysis source: {source}")
