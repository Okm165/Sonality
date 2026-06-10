"""Semantic feature extraction and storage (personality, preferences, knowledge, relationships).

Fully async: LLM calls via ``sonality.caller.async_llm_call`` (gated by ``_llm_gate``),
embeddings via ``sonality.caller.async_embed_documents`` (gated by ``_embedding_gate``),
Qdrant via ``AsyncQdrantClient``.  No threads needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from shared.embedder import Embedder
from shared.llm.parse import normalize_llm_list_response
from shared.types import deterministic_id

from .. import config
from ..caller import async_embed_documents, async_llm_call, format_prompt
from ..prompts import FEATURE_CONSOLIDATION_PROMPT, FEATURE_EXTRACTION_PROMPT, FEATURE_TAGS
from ..schema import DENSE_VECTOR, Collection, SemanticCategory

log = structlog.get_logger(__name__)

SEMANTIC_CATEGORIES: list[SemanticCategory] = list(SemanticCategory)
CONSOLIDATION_THRESHOLD: int = 20
_MAX_CITATIONS: int = 50


class FeatureCommandType(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class FeatureConsolidationDecision(StrEnum):
    CONSOLIDATE = "CONSOLIDATE"
    SKIP = "SKIP"


_VALID_FEATURE_TAGS: frozenset[str] = frozenset(
    tag.strip().lower().replace(" ", "_")
    for tags_str in FEATURE_TAGS.values()
    for tag in tags_str.split(",")
)


class FeatureCommand(BaseModel):
    command: FeatureCommandType
    tag: str = ""
    feature: str = Field(max_length=150)
    value: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> object:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return max(0.0, min(1.0, float(v)))
        return v

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v

    @field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, v: object) -> str:
        return v.strip().lower().replace(" ", "_") if isinstance(v, str) else ""

    @field_validator("feature", "reason", mode="before")
    @classmethod
    def truncate_text(cls, v: object) -> str:
        return str(v)[:150] if v is not None else ""

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: object) -> str:
        if isinstance(v, str):
            return re.sub(r"\s*\(conf=[\d.]+\)\s*$", "", v).strip()[:200]
        if isinstance(v, dict):
            first = next(iter(v.values()), "")
            return str(first)[:200] if isinstance(first, (str, int, float)) else ""
        return ""


class FeatureExtractionResponse(BaseModel):
    commands: list[FeatureCommand] = Field(default_factory=list, max_length=30)

    @model_validator(mode="before")
    @classmethod
    def normalize_commands(cls, data: object) -> object:
        return normalize_llm_list_response(data, list_key="commands", item_required_key="command")

    @model_validator(mode="after")
    def validate_tags(self) -> FeatureExtractionResponse:
        before = len(self.commands)
        self.commands = [cmd for cmd in self.commands if cmd.tag and cmd.tag in _VALID_FEATURE_TAGS]
        dropped = before - len(self.commands)
        if dropped:
            log.debug("feature_extraction_unknown_tags_dropped", dropped_count=dropped)
        return self


class FeatureConsolidationAction(BaseModel):
    source_uid: str = Field(max_length=100)
    target_uid: str = Field(max_length=100)
    canonical_tag: str = Field(default="", max_length=100)
    canonical_feature: str = Field(default="", max_length=150)
    canonical_value: str = ""
    reason: str = Field(default="", max_length=2000)

    @field_validator("canonical_value", mode="before")
    @classmethod
    def truncate_canonical_value(cls, v: object) -> str:
        return str(v)[:500] if v is not None else ""

    @model_validator(mode="before")
    @classmethod
    def normalize_uid_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            for alt, canonical in (
                ("from", "source_uid"),
                ("source_id", "source_uid"),
                ("source", "source_uid"),
                ("to", "target_uid"),
                ("target_id", "target_uid"),
                ("target", "target_uid"),
            ):
                if alt in data and canonical not in data:
                    data[canonical] = data.pop(alt)
        return data


class FeatureConsolidationResponse(BaseModel):
    consolidation_decision: FeatureConsolidationDecision = FeatureConsolidationDecision.SKIP
    reasoning: str = Field(default="", max_length=3000)
    actions: list[FeatureConsolidationAction] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_response(cls, data: object) -> object:
        was_bare_list = isinstance(data, list)
        if isinstance(data, dict):
            cd = data.get("consolidation_decision")
            if isinstance(cd, str):
                data["consolidation_decision"] = cd.strip().upper()
        result = normalize_llm_list_response(
            data, list_key="actions", item_required_key="source_uid"
        )
        if was_bare_list and isinstance(result, dict) and result.get("actions"):
            result.setdefault("consolidation_decision", "CONSOLIDATE")
        return result


@dataclass(frozen=True, slots=True)
class SemanticFeatureRow:
    uid: str
    tag: str
    feature_name: str
    value: str
    confidence: float


class SemanticFeatureExtractor:
    """Fully async semantic feature extraction.

    LLM calls go through ``async_llm_call`` (gated by sonality's semaphore),
    embeddings through ``async_embed_documents`` (gated by embedding semaphore),
    Qdrant through ``AsyncQdrantClient``.  No thread pool slots consumed.
    """

    def __init__(self, qdrant: AsyncQdrantClient, embedder: Embedder) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        log.info("semantic_extractor_ready")

    async def process_episode(
        self,
        episode_uid: str,
        content: str,
        categories: tuple[SemanticCategory, ...] = (),
    ) -> None:
        """Extract features from episode content for specified categories (or all)."""
        target_cats = list(categories) if categories else SEMANTIC_CATEGORIES
        for category in target_cats:
            try:
                await self._extract_features(episode_uid, content, category)
            except Exception:
                log.error(
                    "feature_extraction_failed",
                    episode_uid=episode_uid[:8],
                    category=category,
                    exc_info=True,
                )

    async def _extract_features(
        self, episode_uid: str, content: str, category: SemanticCategory
    ) -> None:
        """Extract features for a single category."""
        existing = await self._load_existing_features(category)
        tags = FEATURE_TAGS.get(category, "")
        result = await async_llm_call(
            instructions=format_prompt(
                FEATURE_EXTRACTION_PROMPT,
                category=category,
                episode_content=content,
                tags=tags,
                existing_features=existing,
            ),
            response_model=FeatureExtractionResponse,
            fallback=FeatureExtractionResponse(),
            model=config.settings.structured_model,
        )
        if not result.success:
            log.debug("feature_extraction_llm_failed", error=(result.error or "")[:80])
            return

        response = result.value

        seen_keys: dict[tuple[str, str], int] = {}
        for i, cmd in enumerate(response.commands):
            key = (cmd.tag or "", cmd.feature or "")
            if key in seen_keys:
                continue
            seen_keys[key] = i
        if len(seen_keys) < len(response.commands):
            dropped = len(response.commands) - len(seen_keys)
            log.debug("feature_extraction_exact_dupes_removed", dropped=dropped)
            response.commands = [response.commands[i] for i in sorted(seen_keys.values())]

        upsert_indices = [
            i
            for i, cmd in enumerate(response.commands)
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}
        ]
        upsert_texts = [
            response.commands[i].value or response.commands[i].feature for i in upsert_indices
        ]
        batch_embeddings: list[list[float]] = []
        if upsert_texts:
            try:
                batch_embeddings = await async_embed_documents(self._embedder, upsert_texts)
            except Exception:
                log.warning(
                    "batch_embedding_failed",
                    feature_count=len(upsert_texts),
                    category=category,
                    exc_info=True,
                )
                return
        embedding_map: dict[int, list[float]] = dict(
            zip(upsert_indices, batch_embeddings, strict=True)
        )

        for i, cmd in enumerate(response.commands):
            is_upsert = cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}
            is_delete = cmd.command is FeatureCommandType.DELETE
            if not (is_upsert or is_delete):
                continue

            if is_upsert:
                log.info(
                    "feature_upsert",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                    value=(cmd.value or "")[:80],
                    confidence=cmd.confidence,
                )
            else:
                log.info(
                    "feature_delete",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                    reason=(cmd.reason or "")[:80],
                )

            embedding = embedding_map.get(i, [])
            if is_upsert and not embedding:
                log.warning(
                    "embedding_missing_for_feature",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                )
                continue

            try:
                await self._persist_command(episode_uid, category, cmd, embedding)
            except Exception:
                log.warning(
                    "feature_persist_failed",
                    episode=episode_uid[:8],
                    category=category,
                    exc_info=True,
                )

        if upsert_indices:
            feature_count = await self._count_features(category)
            if feature_count > CONSOLIDATION_THRESHOLD:
                try:
                    await self._consolidate_features(category)
                except Exception:
                    log.warning("consolidation_failed", category=category, exc_info=True)

    async def _load_existing_features(self, category: SemanticCategory) -> str:
        """Load existing features for context in extraction prompt."""
        try:
            rows = await self._load_feature_rows(category, limit=30)
        except Exception:
            log.debug("semantic_features_load_failed", exc_info=True)
            return "None yet"
        if not rows:
            return "None yet"
        return "\n".join(
            f"- [{row.tag}] {row.feature_name}: {row.value} (conf={row.confidence:.2f})"
            for row in rows
        )

    async def _count_features(self, category: SemanticCategory) -> int:
        """Count features in category for lazy consolidation check."""
        try:
            result = await self._qdrant.count(
                collection_name=Collection.SEMANTIC_FEATURES,
                count_filter=Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=category))]
                ),
            )
            return result.count
        except Exception:
            log.warning("count_features_failed", category=category, exc_info=True)
            return 0

    async def _consolidate_features(self, category: SemanticCategory) -> None:
        """Consolidate duplicate/overlapping features via LLM."""
        features = await self._load_feature_rows(category, limit=40)
        if len(features) < 2:
            return

        log.debug("consolidating_semantic_features", feature_count=len(features), category=category)
        features_text = "\n".join(
            f"[{f.uid[:8]}] [{f.tag}] {f.feature_name}: {f.value} (conf={f.confidence:.2f})"
            for f in features
        )
        result = await async_llm_call(
            instructions=format_prompt(
                FEATURE_CONSOLIDATION_PROMPT, category=category, features=features_text
            ),
            response_model=FeatureConsolidationResponse,
            fallback=FeatureConsolidationResponse(),
            model=config.settings.structured_model,
        )
        if not result.success:
            log.debug("feature_consolidation_llm_failed", error=(result.error or "")[:80])
            return

        response = result.value
        if response.consolidation_decision is FeatureConsolidationDecision.SKIP:
            log.debug("consolidation_skipped", reasoning=response.reasoning[:120])
            return

        log.info(
            "consolidation_started",
            action_count=len(response.actions),
            reasoning=response.reasoning[:120],
        )
        prefix_to_uid = {f.uid[:8]: f.uid for f in features}
        for action in response.actions:
            src = prefix_to_uid.get(action.source_uid, action.source_uid)
            tgt = prefix_to_uid.get(action.target_uid, action.target_uid)
            if not src or not tgt or src == tgt:
                continue
            if src not in prefix_to_uid.values() or tgt not in prefix_to_uid.values():
                continue
            await self._merge_features(category, src, tgt, action)

    async def _load_feature_rows(
        self, category: SemanticCategory, limit: int
    ) -> list[SemanticFeatureRow]:
        """Load feature rows from Qdrant."""
        results, _ = await self._qdrant.scroll(
            collection_name=Collection.SEMANTIC_FEATURES,
            scroll_filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        rows = []
        for point in results:
            if not point.payload:
                continue
            p = point.payload
            rows.append(
                SemanticFeatureRow(
                    uid=str(p.get("uid", "")),
                    tag=str(p.get("tag", "")),
                    feature_name=str(p.get("feature_name", "")),
                    value=str(p.get("value", "")),
                    confidence=float(p.get("confidence") or 0.0),
                )
            )
        rows.sort(key=lambda r: -r.confidence)
        return rows

    async def _merge_features(
        self,
        category: SemanticCategory,
        source_uid: str,
        target_uid: str,
        action: FeatureConsolidationAction,
    ) -> None:
        """Merge source feature into target."""
        results, _ = await self._qdrant.scroll(
            collection_name=Collection.SEMANTIC_FEATURES,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="category", match=MatchValue(value=category)),
                    FieldCondition(key="uid", match=MatchValue(value=target_uid)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            log.warning(
                "merge_target_missing", target_uid=target_uid[:8], source_uid=source_uid[:8]
            )
            return

        target = results[0].payload or {}
        new_value = action.canonical_value or str(target.get("value", ""))

        raw_target_cit = target.get("episode_citations")
        target_citations: list[object] = raw_target_cit if isinstance(raw_target_cit, list) else []

        source_results, _ = await self._qdrant.scroll(
            collection_name=Collection.SEMANTIC_FEATURES,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="category", match=MatchValue(value=category)),
                    FieldCondition(key="uid", match=MatchValue(value=source_uid)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        source_citations: list[object] = []
        if source_results and source_results[0].payload:
            raw_src_cit = source_results[0].payload.get("episode_citations")
            if isinstance(raw_src_cit, list):
                source_citations = raw_src_cit

        merged_citations = list(dict.fromkeys([*target_citations, *source_citations]))[
            -_MAX_CITATIONS:
        ]

        new_payload = {
            "tag": action.canonical_tag or target.get("tag", ""),
            "feature_name": action.canonical_feature or target.get("feature_name", ""),
            "value": new_value,
            "confidence": float(target.get("confidence") or 0.5),
            "episode_citations": merged_citations,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        new_embedding = (await async_embed_documents(self._embedder, [new_value]))[0]
        await self._qdrant.upsert(
            collection_name=Collection.SEMANTIC_FEATURES,
            points=[
                PointStruct(
                    id=target_uid,
                    vector={DENSE_VECTOR: new_embedding},
                    payload={**target, **new_payload},
                )
            ],
        )
        await self._qdrant.delete(
            collection_name=Collection.SEMANTIC_FEATURES,
            points_selector=Filter(
                must=[FieldCondition(key="uid", match=MatchValue(value=source_uid))]
            ),
        )
        log.info(
            "feature_merge",
            target_uid=target_uid[:8],
            source_uid=source_uid[:8],
            reason=action.reason[:80],
        )

    async def _persist_command(
        self,
        episode_uid: str,
        category: SemanticCategory,
        cmd: FeatureCommand,
        embedding: list[float],
    ) -> None:
        """Persist a feature command to Qdrant."""
        seed = f"semantic:{category}:{cmd.tag.strip().lower()}:{cmd.feature.strip().lower()}"
        feature_uid = deterministic_id(seed)
        now = datetime.now(UTC).isoformat()

        if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
            existing, _ = await self._qdrant.scroll(
                collection_name=Collection.SEMANTIC_FEATURES,
                scroll_filter=Filter(
                    must=[FieldCondition(key="uid", match=MatchValue(value=feature_uid))]
                ),
                limit=1,
                with_payload=True,
            )
            citations = [episode_uid]
            created_at = now
            if existing and existing[0].payload:
                old_citations = existing[0].payload.get("episode_citations", [])
                if isinstance(old_citations, list):
                    citations = list(dict.fromkeys([*old_citations, episode_uid]))[-_MAX_CITATIONS:]
                created_at = existing[0].payload.get("created_at", now)

            point = PointStruct(
                id=feature_uid,
                vector={DENSE_VECTOR: embedding},
                payload={
                    "uid": feature_uid,
                    "category": category,
                    "tag": cmd.tag,
                    "feature_name": cmd.feature,
                    "value": cmd.value,
                    "episode_citations": citations,
                    "confidence": max(0.0, min(1.0, cmd.confidence)),
                    "created_at": created_at,
                    "updated_at": now,
                },
            )
            await self._qdrant.upsert(collection_name=Collection.SEMANTIC_FEATURES, points=[point])
        elif cmd.command is FeatureCommandType.DELETE:
            await self._qdrant.delete(
                collection_name=Collection.SEMANTIC_FEATURES,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value=category)),
                        FieldCondition(key="tag", match=MatchValue(value=cmd.tag)),
                        FieldCondition(key="feature_name", match=MatchValue(value=cmd.feature)),
                    ]
                ),
            )
