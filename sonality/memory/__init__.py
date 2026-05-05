"""Public API for the memory subsystem.

Re-exports from:
  graph        — Neo4j episode/belief/snapshot storage
  dual_store   — Neo4j + Qdrant transactional episode lifecycle
  embedder     — text embedding via sentence-transformers
  retrieval/   — LLM-driven query routing, chain/split retrieval, reranking
  segmentation — conversation boundary detection
  semantic_features — background feature extraction worker
"""

from __future__ import annotations

from .belief_provenance import assess_belief_evidence_batch
from .db import DatabaseConnections
from .dual_store import DualEpisodeStore, StoredEpisode
from .embedder import Embedder
from .graph import MemoryGraph
from .knowledge_extract import extract_and_store_knowledge, retrieve_relevant_knowledge
from .retrieval import (
    QueryCategory,
    RoutingDecision,
    SemanticMemoryDecision,
    TemporalExpansionDecision,
    chain_retrieve,
    rerank_episodes,
    route_query,
    split_retrieve,
)
from .segmentation import BoundaryDecision, EventBoundaryDetector
from .semantic_features import SemanticIngestionWorker

__all__ = [
    "BoundaryDecision",
    "DatabaseConnections",
    "DualEpisodeStore",
    "Embedder",
    "EventBoundaryDetector",
    "MemoryGraph",
    "QueryCategory",
    "RoutingDecision",
    "SemanticIngestionWorker",
    "SemanticMemoryDecision",
    "StoredEpisode",
    "TemporalExpansionDecision",
    "assess_belief_evidence_batch",
    "chain_retrieve",
    "extract_and_store_knowledge",
    "rerank_episodes",
    "retrieve_relevant_knowledge",
    "route_query",
    "split_retrieve",
]
