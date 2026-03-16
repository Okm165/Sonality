from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from sonality import config
from sonality.memory.derivatives import DerivativeChunker
from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.embedder import ExternalEmbedder
from sonality.memory.graph import MemoryGraph

if TYPE_CHECKING:
    from tests.containers import ContainerConfig


class _UnusedChunker:
    def chunk_and_embed(self, text: str, episode_uid: str) -> list[object]:
        _ = (text, episode_uid)
        raise AssertionError("chunker should not be called in consistency test")


class _UnusedEmbedder:
    def embed_query(self, query: str) -> list[float]:
        _ = query
        raise AssertionError("embedder should not be called in consistency test")


def test_verify_consistency_cleans_orphans(isolated_both: ContainerConfig) -> None:
    async def _run() -> None:
        uid_a = "11111111-1111-1111-1111-111111111111"
        uid_b = "22222222-2222-2222-2222-222222222222"
        uid_c = "33333333-3333-3333-3333-333333333333"

        qdrant = AsyncQdrantClient(url=isolated_both.qdrant_url)
        neo4j = AsyncGraphDatabase.driver(
            isolated_both.neo4j_url,
            auth=(isolated_both.neo4j_user, isolated_both.neo4j_password),
        )
        try:
            graph = MemoryGraph(neo4j)
            now = datetime.now(UTC).isoformat()
            vector = [0.0] * config.EMBEDDING_DIMENSIONS
            await qdrant.upsert(
                collection_name="derivatives",
                points=[
                    PointStruct(
                        id=uid_a,
                        vector=vector,
                        payload={
                            "uid": uid_a,
                            "episode_uid": "ep-a",
                            "text": "derivative a",
                            "key_concept": "a",
                            "sequence_num": 0,
                            "archived": False,
                            "created_at": now,
                        },
                    ),
                    PointStruct(
                        id=uid_b,
                        vector=vector,
                        payload={
                            "uid": uid_b,
                            "episode_uid": "ep-b",
                            "text": "derivative b",
                            "key_concept": "b",
                            "sequence_num": 1,
                            "archived": False,
                            "created_at": now,
                        },
                    ),
                ],
            )
            async with neo4j.session(database=config.NEO4J_DATABASE) as session:
                await session.run("CREATE (:Derivative {uid: $uid})", uid=uid_b)
                await session.run("CREATE (:Derivative {uid: $uid})", uid=uid_c)

            store = DualEpisodeStore(
                graph=graph,
                qdrant=qdrant,
                chunker=cast(DerivativeChunker, _UnusedChunker()),
                embedder=cast(ExternalEmbedder, _UnusedEmbedder()),
            )
            orphans = await store.verify_consistency()
            assert set(orphans) == {uid_a, uid_c}

            qdrant_points = await qdrant.retrieve(
                collection_name="derivatives",
                ids=[uid_a, uid_b],
                with_payload=True,
            )
            qdrant_ids = {str(point.id) for point in qdrant_points if point.id is not None}
            assert qdrant_ids == {uid_b}

            async with neo4j.session(database=config.NEO4J_DATABASE) as session:
                result = await session.run("MATCH (d:Derivative) RETURN d.uid AS uid")
                neo4j_ids = {str(record["uid"]) async for record in result}
            assert neo4j_ids == {uid_b}
        finally:
            await qdrant.close()
            await neo4j.close()

    asyncio.run(_run())
