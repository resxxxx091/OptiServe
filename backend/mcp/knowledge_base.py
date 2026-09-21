"""
RAG 知识库 —— 基于 Milvus 混合索引的检索实现。

功能：
  1. 文档导入：将文本切片后写入 Milvus，每条切片同时落稠密向量和 bge-m3 稀疏向量
  2. 混合索引召回：一次 query 向量编码 + 两路 ANN 检索（稠密 COSINE / 稀疏 IP），
     两路各自返回有序列表，交给上层做 RRF 融合粗排
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

Milvus 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。

写入失败向上抛（调用方要能告诉用户导入没成功），检索失败也向上抛，
交给 MCP 工具层的熔断与 fallback 处理。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

from pymilvus import DataType, FieldSchema

from core.vector_store import (
    VECTOR_FIELD,
    AsyncEmbeddingClient,
    CollectionSpec,
    MilvusStore,
    VectorStoreConfig,
    VectorStoreError,
    sanitize_text,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_COLLECTIONS = (
    CollectionSpec(
        name="knowledge_base",
        description="OptiServe RAG 知识库（稠密 + 稀疏混合索引）",
        fields=[
            FieldSchema("title", DataType.VARCHAR, max_length=512),
            FieldSchema("content", DataType.VARCHAR, max_length=65535),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("total_chunks", DataType.INT64),
        ],
        # 混合索引的第二路：外部 bge-m3 生成的词表稀疏向量，索引度量 IP
        sparse_fields=("sparse_vector",),
    ),
)


class KnowledgeBase:
    """基于 Milvus 混合索引的 RAG 知识库（向量由外部 embedding 服务生成）。"""

    COLLECTION_NAME = "knowledge_base"
    SPARSE_FIELD = "sparse_vector"
    CHUNK_SIZE = 500
    # 单路召回条数上限：RRF 是排名融合，两路各取一份即可，再多只是浪费 ANN 打分
    MAX_RECALL_PER_PATH = 50
    _OUTPUT_FIELDS = ["title", "content", "chunk_index"]

    def __init__(self, vector_config: Optional[VectorStoreConfig] = None):
        self._vector_config = vector_config or VectorStoreConfig.from_env()
        self._embedder = AsyncEmbeddingClient(self._vector_config)
        self._store = MilvusStore(self._vector_config, KNOWLEDGE_COLLECTIONS)

    async def start(self) -> bool:
        """确保 collection 已就绪；知识库为空时播种默认文档。失败返回 False，由启动闸门决定去留。"""
        if not await self._store.ensure_ready():
            return False

        if await self.doc_count_async() == 0:
            await self._load_default_docs()
        return True

    async def close(self) -> None:
        await self._store.aclose()
        await self._embedder.aclose()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字），稠密 + 稀疏两路向量批量生成后按确定性主键 upsert，
        同一份内容重复导入会覆盖而不是堆积。混合索引要求两路向量同批写入，
        所以 embedding 服务必须同时给得出稀疏那一路，否则整批导入直接抛错。
        """
        rows: List[Dict[str, Any]] = []
        chunks: List[str] = []

        for doc in documents:
            title   = sanitize_text(doc.get("title", ""))
            content = sanitize_text(doc.get("content", ""))
            pieces  = self._chunk_text(content, chunk_size=self.CHUNK_SIZE)

            for i, chunk in enumerate(pieces):
                rows.append({
                    "id":           hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest(),
                    "title":        title,
                    "content":      chunk,
                    "chunk_index":  i,
                    "total_chunks": len(pieces),
                })
                chunks.append(chunk)

        if not rows:
            return 0

        client = await self._store.client()
        if client is None:
            raise VectorStoreError("Milvus 不可用，知识库写入失败")

        vectors = await self._embedder.embed_documents_hybrid(chunks)
        await client.upsert(
            collection_name=self.COLLECTION_NAME,
            data=[
                dict(row, vector=dense, **{self.SPARSE_FIELD: sparse})
                for row, (dense, sparse) in zip(rows, vectors)
            ],
        )
        logger.info(f"知识库导入 {len(rows)} 个文档片段（稠密 + 稀疏双向量）")
        return len(rows)

    async def recall_async(self, query: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        混合索引召回：一次 query 编码，稠密与稀疏两路 ANN 并行检索。

        返回 {"dense": [...], "sparse": [...]} 两份**各自有序**的列表 —— 这里刻意不融合，
        RRF 要在上层把「多个子查询 × 两路索引」共 2N 份排名一起加权融合，
        在这一层就合并掉的话排名信息就只剩一份了。

        稠密路是 COSINE，distance 即相似度（越大越相关）；稀疏路是 IP，分数没有距离含义。
        """
        query_text = sanitize_text(query).strip()
        if not query_text or limit <= 0:
            return {"dense": [], "sparse": []}

        client = await self._store.client()
        if client is None:
            raise VectorStoreError("Milvus 不可用，知识库检索失败")

        per_path = min(limit, self.MAX_RECALL_PER_PATH)
        dense_vector, sparse_vector = await self._embedder.embed_query_hybrid(query_text)
        dense_hits, sparse_hits = await asyncio.gather(
            client.search(
                collection_name=self.COLLECTION_NAME,
                data=[dense_vector],
                anns_field=VECTOR_FIELD,
                limit=per_path,
                filter="",
                output_fields=self._OUTPUT_FIELDS,
            ),
            client.search(
                collection_name=self.COLLECTION_NAME,
                data=[sparse_vector],
                anns_field=self.SPARSE_FIELD,
                limit=per_path,
                filter="",
                output_fields=self._OUTPUT_FIELDS,
            ),
        )

        return {
            "dense":  self._format_hits(dense_hits, "dense"),
            "sparse": self._format_hits(sparse_hits, "sparse"),
        }

    @staticmethod
    def _format_hits(hits: Any, path: str) -> List[Dict[str, Any]]:
        """把一路 Milvus 命中转成统一形状；主键 id 是 RRF 融合时的去重依据。"""
        rows = hits[0] if hits and isinstance(hits[0], list) else []
        items: List[Dict[str, Any]] = []
        for row in rows:
            entity = row.get("entity") or {}
            items.append({
                "id":     row.get("id"),
                "title":  entity.get("title", ""),
                "content": entity.get("content", ""),
                "score":  round(float(row.get("distance", 0.0)), 4),
                "chunk":  entity.get("chunk_index", 0),
                "path":   path,
            })
        return items

    async def doc_count_async(self) -> int:
        """文档片段总数；Milvus 不可用时返回 0（启动统计不该因此失败）。"""
        client = await self._store.client()
        if client is None:
            return 0

        try:
            rows = await client.query(
                collection_name=self.COLLECTION_NAME,
                filter="",
                output_fields=["count(*)"],
            )
        except Exception as ex:
            logger.warning(f"统计知识库片段数失败: {ex}")
            return 0

        if not rows:
            return 0
        return int(rows[0].get("count(*)", 0))

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(
        self, params: Dict[str, Any], context: Any
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        作为 MCP 工具的 handler 注册：一次调用 = 一个查询的混合索引双路召回。

        这里只做到召回为止。RRF 粗排、Reranker 精排、断崖截断在 MCPToolManager 的检索图里做，
        因为融合必须同时看见「改写出的多个子查询 × 两路索引」的全部排名，
        在本层合并就等于把粗排的输入削成一份。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        limit = int(params.get("top_k", 5) or 5)
        return await self.recall_async(query, limit=limit)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    async def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        try:
            await self.add_documents_async(default_docs)
            logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
        except Exception as ex:
            logger.warning(f"默认知识库播种失败: {ex}")
