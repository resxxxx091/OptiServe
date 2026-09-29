"""
亮点：端到端意图识别

三路融合策略（加权投票）：
  1. LLM 语义理解（权重 0.5）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 0.3）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 0.2）—— 零延迟兜底

三路在 LangGraph 的同一条 super-step 里并发跑（START → {llm ∥ embedding ∥ pattern} → vote），
投票结果置信度低于阈值时降级为 OTHER。
"""
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from core.degradation import Dep, degrade
from core.llm import LLMProvider, message_text
from core.tracing import trace_span, usage_attrs
from core.vector_store import AsyncEmbeddingClient, VectorStoreConfig

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    DATA_QUERY        = "data_query"              # 数据查询
    REPORT_GENERATION = "report_generation"       # 报表生成
    PRODUCT_OPS       = "product_ops"             # 商品管理
    CAMPAIGN_OPS      = "campaign_ops"            # 营销活动
    ORDER_OPS         = "order_ops"               # 订单处理
    CS_ESCALATION     = "cs_escalation"           # 客诉处理
    PLATFORM_RULES    = "platform_rules"          # 平台规则咨询
    SOP_HOWTO         = "sop_howto"               # 操作指引
    ANOMALY_DIAGNOSIS = "anomaly_diagnosis"       # 异常诊断
    CONTENT_GENERATE  = "content_generate"        # 内容生成
    SYSTEM_ISSUE      = "system_issue"            # 系统故障
    OTHER      = "other"


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── 模板语料（Embedding 匹配用）───────────────────────────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.DATA_QUERY:   ["上周的销售额是多少", "查一下A商品的库存", "昨天退款率多少", "这个月GMV怎么样"],
    IntentCategory.REPORT_GENERATION: ["帮我生成上周的销售周报", "出一份大促复盘报告", "把这个月的数据导出成报表"],
    IntentCategory.PRODUCT_OPS:  ["把这个商品下架", "帮我改一下价格", "新品什么时候能上架"],
    IntentCategory.CAMPAIGN_OPS: ["配置一张满减优惠券", "怎么报名618活动", "这个月有什么促销活动可以参加"],
    IntentCategory.ORDER_OPS:    ["这单退款帮我审核一下", "有个异常订单要处理", "买家要改收货地址怎么操作"],
    IntentCategory.CS_ESCALATION: ["这个客诉工单帮我升级一下", "消费者投诉了帮我跟进处理", "这个纠纷单催一下处理进度"],
    IntentCategory.PLATFORM_RULES: ["延迟发货会被扣多少分", "虚假宣传有什么处罚", "报名活动需要什么资质"],
    IntentCategory.SOP_HOWTO:    ["后台怎么批量修改运费模板", "设置优惠券的操作步骤是什么", "怎么在后台导出订单明细"],
    IntentCategory.ANOMALY_DIAGNOSIS: ["转化率为什么突然跌了", "店铺流量异常怎么排查", "销量下滑是什么原因"],
    IntentCategory.CONTENT_GENERATE: ["帮我写一个新品上架的卖点文案", "生成一条店铺公告", "给这个活动写句宣传语"],
    IntentCategory.SYSTEM_ISSUE: ["商家后台一直报500错误", "账号登录不上了", "系统崩了没法操作"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ── G3：三路投票图 ─────────────────────────────────────────────────────────────


class IntentState(TypedDict, total=False):
    message: str
    history: Optional[List[Dict[str, str]]]
    llm: Dict[str, Any]
    embedding: Dict[str, Any]
    pattern: Dict[str, Any]
    result: "IntentResult"


def _recognizer(config) -> "IntentRecognizer":
    return config["configurable"]["recognizer"]


async def llm_route(state: IntentState, config) -> Dict[str, Any]:
    return {"llm": await _recognizer(config)._llm_recognize(state["message"], state["history"])}


async def embedding_route(state: IntentState, config) -> Dict[str, Any]:
    return {"embedding": await _recognizer(config)._embedding_recognize(state["message"])}


async def pattern_route(state: IntentState, config) -> Dict[str, Any]:
    return {"pattern": _recognizer(config)._pattern_recognize(state["message"])}


async def vote_route(state: IntentState, config) -> Dict[str, Any]:
    rec = _recognizer(config)
    intent, confidence, source_scores = rec._vote(state["llm"], state["embedding"], state["pattern"])
    return {
        "result": IntentResult(
            intent=intent,
            confidence=confidence,
            entities=rec._extract_entities(state["message"]),
            source_scores=source_scores,
        )
    }


def build_intent_graph():
    graph = StateGraph(IntentState)
    graph.add_node("llm", llm_route)
    graph.add_node("embedding", embedding_route)
    graph.add_node("pattern", pattern_route)
    graph.add_node("vote", vote_route)

    for name in ("llm", "embedding", "pattern"):
        graph.add_edge(START, name)
        graph.add_edge(name, "vote")
    graph.add_edge("vote", END)
    return graph.compile()


INTENT_GRAPH = build_intent_graph()


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 langchain-deepseek 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        *,
        model: str,
        confidence_threshold: float = 0.5,
        vector_config: Optional[VectorStoreConfig] = None,
    ):
        self._llm      = LLMProvider(api_key, base_url)
        self.model     = model
        self.threshold = confidence_threshold
        self._embedder = AsyncEmbeddingClient(vector_config or VectorStoreConfig.from_env())

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            return self._cache[key]

        state = await INTENT_GRAPH.ainvoke(
            {"message": message, "history": history},
            {"configurable": {"recognizer": self}},
        )
        result = state["result"]

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（示例写在 prompt 里 + 最近对话上下文）。"""
        message = self._clean_text(message)
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是电商运营服务平台的意图分析专家。提问的是商家/运营人员，根据示例判断其意图，返回 JSON。
请从可选意图中选出最贴合用户问题的一个。
例如查销售额用 data_query，配置优惠券用 campaign_ops，问扣分处罚用 platform_rules，后台报错用 system_issue。

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            chat = self._llm.chat_model(model=self.model, temperature=0.1, max_tokens=256)
            with trace_span("llm_intent", input=prompt, model=self.model) as span:
                # 模型名走 OTel 属性：Langfuse 成本面板按 gen_ai.request.model 查价格表
                span.otel = {"gen_ai.request.model": self.model}
                resp = await chat.ainvoke([HumanMessage(content=prompt)])
                raw = message_text(resp)
                span.output = raw
                span.otel.update(usage_attrs(resp))
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            degrade(Dep.LLM, "intent_llm_failed", f"LLM 意图识别失败，该路记 0 分: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败"}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embedder.embed_query(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            degrade(Dep.EMBEDDING, "intent_embedding_failed", f"Embedding 意图识别失败，该路记 0 分: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        patterns = {
            IntentCategory.DATA_QUERY:   ["销售额", "退款率", "转化率", "库存还有", "查一下数据", "gmv", "访客", "销量多少", "uv"],
            IntentCategory.REPORT_GENERATION: ["周报", "日报", "报表", "复盘", "导出报告", "月度总结", "经营分析"],
            IntentCategory.PRODUCT_OPS:  ["下架", "上架", "改价", "调库存", "商品信息修改", "sku", "批量修改商品"],
            IntentCategory.CAMPAIGN_OPS: ["优惠券", "满减", "活动报名", "促销活动", "秒杀", "营销投放"],
            IntentCategory.ORDER_OPS:    ["退款审核", "退款处理", "退款", "异常订单", "改收货地址", "订单处理", "拦截订单", "订单号"],
            IntentCategory.CS_ESCALATION: ["客诉", "工单", "投诉", "升级处理", "催办", "纠纷单"],
            IntentCategory.PLATFORM_RULES: ["扣几分", "扣分", "处罚", "违规", "资质", "保证金", "平台规则", "入驻", "延迟发货", "罚款"],
            IntentCategory.SOP_HOWTO:    ["怎么操作", "怎么设置", "怎么配置", "操作步骤", "后台怎么", "如何导出", "怎么开通"],
            IntentCategory.ANOMALY_DIAGNOSIS: ["为什么跌", "为什么下降", "异常", "排查", "流量掉了", "波动", "下滑", "跌", "突然跌", "掉了"],
            IntentCategory.CONTENT_GENERATE: ["写一个", "写一条", "文案", "公告", "宣传语", "帮我想", "生成一段"],
            IntentCategory.SYSTEM_ISSUE: ["报错", "500", "崩溃", "登录不上", "系统故障", "bug", "白屏", "无法访问"],
        }
        best_cat, best_score = self._best_pattern_match(msg, patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        scores: Dict[IntentCategory, float] = {}
        for result, w in ((llm, 0.5), (emb, 0.3), (pat, 0.2)):
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best]
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(
                    r"(?:error(?:_code)?|错误码|状态码|http|返回|出现|报错|报了|报)[^0-9，。；！？、\n]{0,6}?([45]\d{2})(?!\d)",
                    message, re.I,
                )
            ),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = await self._embedder.embed_documents(all_texts)
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload: Dict[str, Any] = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")
