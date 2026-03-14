from __future__ import annotations

import asyncio
from typing import cast

from psycopg_pool import AsyncConnectionPool

from sonality.memory.derivatives import DerivativeChunker
from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.embedder import ExternalEmbedder
from sonality.memory.graph import MemoryGraph


class _FakeCursor:
    def __init__(self, select_rows: list[tuple[str]]) -> None:
        self._select_rows = select_rows
        self.deletes: list[list[str]] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def execute(self, query: str, params: tuple[list[str], ...] = ()) -> None:
        if query.strip().startswith("DELETE FROM derivatives") and params:
            self.deletes.append(params[0])

    async def fetchall(self) -> list[tuple[str]]:
        return self._select_rows


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    async def __aenter__(self) -> _FakeConnection:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return self._cursor


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connection(self) -> _FakeConnection:
        return _FakeConnection(self._cursor)


class _FakeGraph:
    def __init__(self, derivative_uids: set[str]) -> None:
        self._derivative_uids = derivative_uids
        self.deleted_neo4j: list[list[str]] = []

    async def list_derivative_uids(self) -> set[str]:
        return set(self._derivative_uids)

    async def delete_derivatives(self, uids: list[str]) -> None:
        self.deleted_neo4j.append(list(uids))


class _UnusedChunker:
    def chunk_and_embed(self, text: str, episode_uid: str) -> list[object]:
        _ = (text, episode_uid)
        raise AssertionError("chunker should not be called in consistency test")


class _UnusedEmbedder:
    def embed_query(self, query: str) -> list[float]:
        _ = query
        raise AssertionError("embedder should not be called in consistency test")


def test_verify_consistency_cleans_orphans() -> None:
    cursor = _FakeCursor([("d-a",), ("d-b",)])
    graph = _FakeGraph({"d-b", "d-c"})
    store = DualEpisodeStore(
        graph=cast(MemoryGraph, graph),
        pg_pool=cast(AsyncConnectionPool, _FakePool(cursor)),
        chunker=cast(DerivativeChunker, _UnusedChunker()),
        embedder=cast(ExternalEmbedder, _UnusedEmbedder()),
    )
    orphans = asyncio.run(store.verify_consistency())
    assert set(orphans) == {"d-a", "d-c"}
    assert cursor.deletes == [["d-a"]]
    assert graph.deleted_neo4j == [["d-c"]]
