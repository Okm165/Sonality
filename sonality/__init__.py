"""Sonality — LLM agent with a self-evolving personality."""

import logging

__version__ = "0.1.0"

# Configure library loggers early to prevent debug spam.
# These libraries emit excessive DEBUG/INFO logs that obscure application logs.
for _lib in ("httpcore", "httpx", "neo4j", "neo4j.io", "neo4j.pool"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
# Neo4j notifications about missing properties are benign (empty database state).
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
