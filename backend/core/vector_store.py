"""
向量层共享组件：Milvus 异步客户端 + 外部 embedding（稠密 + 稀疏）+ 外部 reranker + collection 脚手架。

Milvus 只负责向量的存取与检索，不生成向量，所以向量必须由调用方产出。这里把几件共用的事
收敛到一处，供 memory/conversation_memory.py 与 mcp/knowledge_base.py 复用：

  1. AsyncEmbeddingClient —— OpenAI 兼容 /v1/embeddings 的异步客户端（httpx 原生异步），
     同一次请求可以拿回稠密向量和 bge-m3 的词表稀疏向量
  2. AsyncRerankClient —— Cohere/Jina 风格 /v1/rerank 的异步客户端，精排用交叉编码器而不是 LLM
  3. Milvus 的客户端创建、schema 构造、collection 建/校验（含稀疏向量字段的索引）

Milvus 表达式没有参数绑定，凡是拼 user_id / conv_id 这类外部输入，都必须过 milvus_literal()。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from pymilvus import AsyncMilvusClient, CollectionSchema, DataType, FieldSchema
from pymilvus.client.types import ConsistencyLevel

logger = logging.getLogger(__name__)

VECTOR_FIELD = "vector"
DEFAULT_METRIC_TYPE = "COSINE"
# 稀疏向量是外部模型产出的词权重（不是 Milvus 自建 BM25 统计），相似度语义是内积，没有距离含义。
SPARSE_METRIC_TYPE = "IP"
# Strong：压缩后立刻检索、画像 upsert 后立刻读回都要求 read-your-writes；数据量很小，代价可忽略。
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.Strong

# 请求稀疏向量时塞进请求体的开关字段名（各家 OpenAI 兼容服务的私有扩展，如 SiliconFlow 的 return_sparse）
SPARSE_REQUEST_PARAM = "return_sparse"

# bge-m3 的稠密向量维度。换不同维度的 embedding 模型时改这里，并先删除已建好的 collection 重建。
EMBEDDING_DIM = 1024

# 一次响应里稀疏向量可能出现的键名与形状，各家 OpenAI 兼容服务的叫法不统一
_SPARSE_KEYS = ("sparse", "sparse_embedding", "sparse_vector", "lexical_weights")

SparseVector = Dict[int, float]


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning(f"忽略非法浮点配置 {name}={os.getenv(name)!r}")
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning(f"忽略非法整数配置 {name}={os.getenv(name)!r}")
        return default


class VectorStoreError(Exception):
    """向量层不可用：Milvus 连不上、collection 维度不一致、embedding 调用失败等。"""


class EmbeddingError(VectorStoreError):
    """embedding 服务不可用或返回结果不合法。"""


class RerankError(VectorStoreError):
    """rerank 服务不可用或返回结果不合法。"""


@dataclass(frozen=True)
class VectorStoreConfig:
    """Milvus、embedding 与 rerank 的连接配置，默认值面向本机开发，部署时由环境变量覆盖。"""

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    milvus_timeout_s: float = 10.0
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "bge-m3"
    embedding_timeout_s: float = 15.0
    embedding_batch: int = 32
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "bge-reranker-v2-m3"
    # 定死在代码里：这一跳不是评测调参项，只要保证小于 RETRIEVAL_TOTAL_TIMEOUT_S 即可
    rerank_timeout_s: float = 10.0

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        return cls(
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            milvus_token=os.getenv("MILVUS_TOKEN", ""),
            milvus_db_name=os.getenv("MILVUS_DB_NAME", "default"),
            milvus_timeout_s=_env_float("MILVUS_TIMEOUT_S", 10.0),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            embedding_timeout_s=_env_float("EMBEDDING_TIMEOUT_S", 15.0),
            embedding_batch=max(1, _env_int("EMBEDDING_BATCH", 32)),
            rerank_base_url=os.getenv("RERANK_BASE_URL", "").strip(),
            # 多数托管平台 embedding 与 rerank 同一账号同一密钥，不单独配时跟着 embedding 走
            rerank_api_key=(os.getenv("RERANK_API_KEY") or os.getenv("EMBEDDING_API_KEY", "")).strip(),
            rerank_model=os.getenv("RERANK_MODEL", "bge-reranker-v2-m3"),
        )


def sanitize_text(value: Any) -> str:
    """转成可安全送进 HTTP / Milvus 的 UTF-8 字符串（剥离非法代理字符）。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="ignore").decode("utf-8")


def milvus_literal(value: Any) -> str:
    """把值转成 Milvus 过滤表达式里的字符串字面量。

    Milvus 表达式不支持参数绑定，user_id / conv_id 又来自 API 入参，
    直接 f'user_id == "{v}"' 会被引号或反斜杠打断（甚至改变过滤语义），故统一走 json 转义。
    """
    return json.dumps(sanitize_text(value), ensure_ascii=False)


def build_schema(
    dim: int,
    extra_fields: Sequence[FieldSchema],
    description: str = "",
    sparse_fields: Sequence[str] = (),
) -> CollectionSchema:
    """构造 collection schema：主键 id + 稠密向量字段由这里统一加，业务字段由调用方传入。

    sparse_fields 声明的稀疏向量字段也在这里补 —— 混合索引的两路向量必须同源建表，
    让调用方各写一份 FieldSchema 只会造成形状漂移。
    """
    fields = [
        FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64, description="主键"),
        FieldSchema(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim, description="外部 embedding 生成的稠密向量"),
        *[FieldSchema(name, DataType.SPARSE_FLOAT_VECTOR) for name in sparse_fields],
        *extra_fields,
    ]
    return CollectionSchema(fields, description=description, enable_dynamic_field=False)


def create_async_client(config: VectorStoreConfig) -> AsyncMilvusClient:
    """创建 Milvus 异步客户端（构造函数不建立连接，首次调用才真正连）。"""
    return AsyncMilvusClient(
        uri=config.milvus_uri,
        token=config.milvus_token or "",
        db_name=config.milvus_db_name or "default",
        timeout=config.milvus_timeout_s,
    )


def vector_dim(description: Dict[str, Any]) -> Optional[int]:
    """从 describe_collection 的结果里取出稠密向量维度（只有稠密向量字段带 dim 参数）。"""
    for field in description.get("fields") or []:
        params = field.get("params") or {}
        if "dim" in params:
            try:
                return int(params["dim"])
            except (TypeError, ValueError):
                return None
    return None


def field_names(description: Dict[str, Any]) -> set:
    """collection 现有字段名集合，用来判断老 collection 缺不缺稀疏向量字段。"""
    return {field.get("name") for field in (description.get("fields") or [])}


async def ensure_collection(
    client: AsyncMilvusClient,
    collection_name: str,
    schema: CollectionSchema,
    dim: int,
    sparse_fields: Sequence[str] = (),
) -> None:
    """幂等建 collection：不存在则建（含索引并自动 load），已存在则校验维度与稀疏字段。

    对不上时抛 VectorStoreError 而不是自动重建：换 embedding 模型（维度变）或补混合索引
    （稀疏字段是建表后加不出来的）都必须人工确认后再删库，否则会静默丢掉已有向量数据。
    """
    if await client.has_collection(collection_name):
        description = await client.describe_collection(collection_name)
        actual = vector_dim(description)
        if actual is not None and actual != dim:
            raise VectorStoreError(
                f"collection {collection_name} 的向量维度是 {actual}，"
                f"与代码里的 EMBEDDING_DIM={dim} 不一致；换 embedding 模型后需先删除该 collection"
            )
        missing = [name for name in sparse_fields if name not in field_names(description)]
        if missing:
            raise VectorStoreError(
                f"collection {collection_name} 缺少稀疏向量字段 {missing}，混合索引检索不了这一路；"
                f"SPARSE_FLOAT_VECTOR 无法在已存在的 collection 上追加，需先删除该 collection 让它重新播种"
            )
        await client.load_collection(collection_name)
        return

    index_params = client.prepare_index_params()
    index_params.add_index(field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type=DEFAULT_METRIC_TYPE)
    for name in sparse_fields:
        index_params.add_index(field_name=name, index_type="AUTOINDEX", metric_type=SPARSE_METRIC_TYPE)
    await client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level=DEFAULT_CONSISTENCY_LEVEL,
    )
    logger.info(
        f"Milvus collection 已创建: {collection_name} "
        f"(dim={dim}, metric={DEFAULT_METRIC_TYPE}, sparse={list(sparse_fields) or '无'})"
    )


@dataclass(frozen=True)
class CollectionSpec:
    """一个 collection 的业务字段定义（主键与向量字段由 build_schema 统一补）。

    sparse_fields 声明该 collection 参与混合检索的稀疏向量字段名。
    """

    name: str
    description: str
    fields: Sequence[FieldSchema]
    sparse_fields: Sequence[str] = ()


class MilvusStore:
    """Milvus 客户端的懒加载外壳：首次调用才连，失败后按冷却期自动重试。

    记忆与知识库各持有一个实例、各自管理自己的 collection；不可用时 client() 返回 None，
    由调用方决定降级方式（检索返回空 / 写入丢弃 / 向上抛）。
    """

    RETRY_COOLDOWN_S = 30.0

    def __init__(self, config: VectorStoreConfig, collections: Sequence[CollectionSpec]):
        self._config = config
        self._collections = list(collections)
        self._client: Optional[AsyncMilvusClient] = None
        self._lock = asyncio.Lock()
        self._retry_at = 0.0
        self._warned = False

    @property
    def config(self) -> VectorStoreConfig:
        return self._config

    async def ensure_ready(self) -> bool:
        """确保客户端可用且所有 collection 已就绪。"""
        if self._client is not None:
            return True

        async with self._lock:
            if self._client is not None:
                return True
            if time.monotonic() < self._retry_at:
                return False

            client: Optional[AsyncMilvusClient] = None
            try:
                client = create_async_client(self._config)
                dim = EMBEDDING_DIM
                for spec in self._collections:
                    await ensure_collection(
                        client,
                        spec.name,
                        build_schema(dim, spec.fields, spec.description, spec.sparse_fields),
                        dim=dim,
                        sparse_fields=spec.sparse_fields,
                    )
            except Exception as ex:
                self._retry_at = time.monotonic() + self.RETRY_COOLDOWN_S
                if self._warned:
                    logger.debug(f"Milvus 仍不可用: {ex}")
                else:
                    logger.warning(
                        f"Milvus 不可用（{ex}）；相关向量功能降级，"
                        f"{self.RETRY_COOLDOWN_S:.0f}s 后自动重试"
                    )
                    self._warned = True
                await self._close_quietly(client)
                return False

            self._client = client
            self._warned = False
            logger.info(
                f"Milvus 已连接: {self._config.milvus_uri} "
                f"(dim={EMBEDDING_DIM}, model={self._config.embedding_model}, "
                f"collections={[spec.name for spec in self._collections]})"
            )
            return True

    async def client(self) -> Optional[AsyncMilvusClient]:
        """拿到可用客户端；不可用返回 None。"""
        if await self.ensure_ready():
            return self._client
        return None

    async def aclose(self) -> None:
        client, self._client = self._client, None
        await self._close_quietly(client)

    @staticmethod
    async def _close_quietly(client: Optional[AsyncMilvusClient]) -> None:
        if client is None:
            return
        try:
            await client.close()
        except Exception as ex:
            logger.debug(f"关闭 Milvus 客户端失败: {ex}")


class AsyncEmbeddingClient:
    """OpenAI 兼容 /v1/embeddings 的异步客户端。

    向量由外部服务生成（`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL`），
    走 httpx 原生异步，不再把同步客户端丢进线程池；一个实例持有一个连接池。

    bge-m3 这类模型同时输出稠密向量和词表稀疏向量，`*_hybrid` 方法一次请求拿回两路，
    稀疏那一路供混合索引的 BM25/关键词语义检索使用。
    """

    def __init__(self, config: VectorStoreConfig):
        self._config = config
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def model(self) -> str:
        return self._config.embedding_model

    def _endpoint(self) -> str:
        base = sanitize_text(self._config.embedding_base_url).strip().rstrip("/")
        if not base:
            raise EmbeddingError("未配置 EMBEDDING_BASE_URL，无法生成向量")
        if not self._config.embedding_api_key:
            raise EmbeddingError("未配置 EMBEDDING_API_KEY，无法生成向量")
        # 允许把完整端点直接写进 EMBEDDING_BASE_URL
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    def _session(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._config.embedding_timeout_s)
        return self._http

    async def embed_documents(
        self, texts: Sequence[str], timeout: Optional[float] = None
    ) -> List[List[float]]:
        """批量生成稠密向量，内部按 embedding_batch 分批，返回顺序与入参一致。

        timeout 覆盖本批请求的超时；不传则用配置里的 embedding_timeout_s。
        在线读路径必须传短值，否则一次慢 embedding 就能拖垮整个请求。
        """
        return [dense for dense, _ in await self._embed(texts, timeout, want_sparse=False)]

    async def embed_query(self, text: str, timeout: Optional[float] = None) -> List[float]:
        vectors = await self.embed_documents([text], timeout=timeout)
        if not vectors:
            raise EmbeddingError("embedding 服务返回空结果")
        return vectors[0]

    async def embed_documents_hybrid(
        self, texts: Sequence[str], timeout: Optional[float] = None
    ) -> List[Tuple[List[float], SparseVector]]:
        """批量生成 (稠密向量, 稀疏向量)，顺序与入参一致。

        服务端不返回稀疏向量时抛 EmbeddingError —— 混合索引没有降级路径，
        少一路就是检索结果不可信，宁可报错。
        """
        return await self._embed(texts, timeout, want_sparse=True)

    async def embed_query_hybrid(
        self, text: str, timeout: Optional[float] = None
    ) -> Tuple[List[float], SparseVector]:
        vectors = await self.embed_documents_hybrid([text], timeout=timeout)
        if not vectors:
            raise EmbeddingError("embedding 服务返回空结果")
        return vectors[0]

    async def _embed(
        self, texts: Sequence[str], timeout: Optional[float], want_sparse: bool
    ) -> List[Tuple[List[float], SparseVector]]:
        cleaned = [sanitize_text(text) for text in texts]
        if not cleaned:
            return []

        batch_size = self._config.embedding_batch
        pairs: List[Tuple[List[float], SparseVector]] = []
        for start in range(0, len(cleaned), batch_size):
            pairs.extend(
                await self._embed_batch(cleaned[start:start + batch_size], timeout, want_sparse)
            )
        return pairs

    async def _embed_batch(
        self, batch: List[str], timeout: Optional[float], want_sparse: bool
    ) -> List[Tuple[List[float], SparseVector]]:
        endpoint = self._endpoint()
        body: Dict[str, Any] = {"model": self._config.embedding_model, "input": batch}
        if want_sparse:
            body[SPARSE_REQUEST_PARAM] = True
        request = {
            "headers": {
                "Authorization": f"Bearer {self._config.embedding_api_key}",
                "Content-Type": "application/json",
            },
            "json": body,
        }
        if timeout is not None:
            request["timeout"] = timeout
        try:
            resp = await self._session().post(endpoint, **request)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as ex:
            raise EmbeddingError(f"embedding 请求失败: {ex}") from ex
        except ValueError as ex:
            raise EmbeddingError(f"embedding 响应不是合法 JSON: {ex}") from ex

        items = payload.get("data") if isinstance(payload, dict) else None
        try:
            # 部分服务不保证顺序，按 index 还原
            ordered = sorted(items or [], key=lambda i: i.get("index", 0))
            vectors = [(list(item["embedding"]), _parse_sparse(item) if want_sparse else {})
                       for item in ordered]
        except (TypeError, KeyError) as ex:
            raise EmbeddingError(f"embedding 响应缺少 data[].embedding: {payload}") from ex

        if len(vectors) != len(batch):
            raise EmbeddingError(f"embedding 返回 {len(vectors)} 条，期望 {len(batch)} 条")
        for dense, sparse in vectors:
            if len(dense) != EMBEDDING_DIM:
                raise EmbeddingError(
                    f"embedding 维度 {len(dense)} 与代码里的 EMBEDDING_DIM={EMBEDDING_DIM} 不一致"
                )
            if want_sparse and not sparse:
                raise EmbeddingError(
                    f"embedding 服务未返回稀疏向量（model={self._config.embedding_model}）；"
                    f"混合索引少一路不可信，若端点的开关字段名不同请改 core/vector_store.py 的 SPARSE_REQUEST_PARAM"
                )
        return vectors

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def _parse_sparse(item: Dict[str, Any]) -> SparseVector:
    """从一条 embedding 响应里取出稀疏向量，归一成 Milvus 要的 {token_id: weight}。

    稀疏那一路的键名各家叫法不一（sparse / sparse_embedding / lexical_weights），值可能是
    {id: weight} 字典、{"indices": [...], "values": [...]} 双数组或 [[id, weight], ...]，
    这里统一收下；零权重和非法项丢掉（稀疏向量体积直接取决于非零项数）。
    """
    for key in _SPARSE_KEYS:
        raw = item.get(key)
        if raw is None:
            continue
        pairs = _sparse_pairs(raw)
        sparse: SparseVector = {}
        for index, weight in pairs:
            try:
                token_id = int(index)
                value = float(weight)
            except (TypeError, ValueError):
                continue
            if value:
                sparse[token_id] = sparse.get(token_id, 0.0) + value
        if sparse:
            return sparse
    return {}


def _sparse_pairs(raw: Any) -> Sequence[Any]:
    if isinstance(raw, dict):
        if isinstance(raw.get("indices"), list) and isinstance(raw.get("values"), list):
            return list(zip(raw["indices"], raw["values"]))
        return list(raw.items())
    if isinstance(raw, list):
        return [tuple(entry) for entry in raw if isinstance(entry, (list, tuple)) and len(entry) == 2]
    return []


class AsyncRerankClient:
    """Cohere/Jina 风格 /v1/rerank 的异步客户端（交叉编码器精排）。

    和 embedding 一样的外部服务形态：httpx 原生异步、一个实例一个连接池。
    精排是把 query 和每个候选文档拼成一对送进模型打分，所以候选数直接决定这一路的耗时，
    上游必须先用 RRF 粗排把候选压到几十条以内。

    调用失败一律抛 RerankError，不做静默兜底：精排是检索质量的最后一道，
    拿召回顺序冒充精排结果，评测数字里看不出区别。
    """

    def __init__(self, config: VectorStoreConfig):
        self._config = config
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def model(self) -> str:
        return self._config.rerank_model

    def _endpoint(self) -> str:
        base = sanitize_text(self._config.rerank_base_url).strip().rstrip("/")
        if not base:
            raise RerankError("未配置 RERANK_BASE_URL，无法精排")
        if not self._config.rerank_api_key:
            raise RerankError("未配置 RERANK_API_KEY，无法精排")
        # 允许把完整端点直接写进 RERANK_BASE_URL
        return base if base.endswith("/rerank") else f"{base}/rerank"

    def _session(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._config.rerank_timeout_s)
        return self._http

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """给 documents 打分，返回 [(原始下标, 相关性分数)]，按分数降序。

        分数是否落在 [0, 1] 取决于服务端有没有做归一化（bge-reranker 系一般给的是
        sigmoid 后的概率）；下游的断崖截断按相对差取阈值，量纲一致即可。
        """
        if not documents:
            return []
        payload: Dict[str, Any] = {
            "model": self._config.rerank_model,
            "query": sanitize_text(query),
            "documents": [sanitize_text(doc) for doc in documents],
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(int(top_n), len(documents)))

        request: Dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {self._config.rerank_api_key}",
                "Content-Type": "application/json",
            },
            "json": payload,
        }
        if timeout is not None:
            request["timeout"] = timeout

        try:
            resp = await self._session().post(self._endpoint(), **request)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as ex:
            raise RerankError(f"rerank 请求失败: {ex}") from ex
        except ValueError as ex:
            raise RerankError(f"rerank 响应不是合法 JSON: {ex}") from ex

        results = body.get("results") if isinstance(body, dict) else None
        if results is None and isinstance(body, dict):
            results = body.get("data")
        if not isinstance(results, list):
            raise RerankError(f"rerank 响应缺少 results[]: {str(body)[:200]}")

        scored: List[Tuple[int, float]] = []
        for entry in results:
            if not isinstance(entry, dict) or "index" not in entry:
                raise RerankError(f"rerank 结果缺少 index: {str(entry)[:200]}")
            index = int(entry["index"])
            if not 0 <= index < len(documents):
                raise RerankError(f"rerank 返回下标越界: {index} / {len(documents)} 条候选")
            raw_score = entry.get("relevance_score", entry.get("score"))
            if raw_score is None:
                raise RerankError(f"rerank 结果缺少 relevance_score: {str(entry)[:200]}")
            scored.append((index, float(raw_score)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
