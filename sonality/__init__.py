"""Sonality — LLM agent with a self-evolving personality.

Configures library log levels at import time so noisy HTTP/Neo4j output
doesn't drown application logs regardless of the root log level.
"""

import logging

__version__ = "0.1.0"

for _lib in ("httpcore", "httpx", "neo4j", "neo4j.io", "neo4j.pool"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
