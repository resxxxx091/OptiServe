"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import json
import logging
import pathlib
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from core.intent_recognizer import IntentRecognizer
from core.llm import LLMProvider, message_text

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """
    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(self, llm: LLMProvider, model: str):
        self._llm    = llm
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            chat = self._llm.chat_model(model=self._model, temperature=0.0, max_tokens=256)
            resp = await chat.ainvoke([HumanMessage(content=prompt)])
            raw = message_text(resp)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程（落在 G6 评测图里，见文件末尾 EVAL_GRAPH）：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
    ):
        llm = LLMProvider(api_key, base_url)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(llm, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测（G6 评测图）。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        state = await EVAL_GRAPH.ainvoke(
            {"intent_cases": intent_cases, "dialog_cases": dialog_cases},
            {"configurable": {"evaluator": self}},
        )
        return state["report"]

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
        )


# ── G6：端到端评测图 ──────────────────────────────────────────────────────────
#
#   START → collect_cases → {intent_eval ∥ dialog_eval} → aggregate → regression
#         → recommendations → report → END
#
# 两条评测分支各占一个节点、并发跑（意图分支只碰识别器，对话分支只碰编排器与裁判），
# 汇合后 aggregate/regression/recommendations/report 保持线性：
#   - report 节点必须排在 regression 之后，因为回归比对读的是 self._history[-1]，
#     历史只在报告生成后才 append，顺序颠倒会让本轮和自身比较。
#   - 对话用例之间仍然逐例串行（节点内部不并发）：多轮用例共享 conv_id 与会话历史，
#     并发会让 /eval/run 的响应顺序和记忆写入次序都发生变化。

class EvalState(TypedDict, total=False):
    intent_cases:    Optional[List[IntentTestCase]]
    dialog_cases:    Optional[List[Dict[str, Any]]]
    intent_metrics:  Dict[str, Any]
    intent_result:   List[EvalResult]
    dialog_results:  List[EvalResult]
    results:         List[EvalResult]
    avg_scores:      Dict[str, float]
    regressions:     List[str]
    recommendations: List[str]
    report:          EvalReport


def _evaluator(config: RunnableConfig) -> "EndToEndEvaluator":
    return config["configurable"]["evaluator"]


async def collect_cases(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    """初始化分支写入的信道：LangGraph 里从未被写的键不会出现在最终状态。"""
    return {"intent_metrics": {}, "intent_result": [], "dialog_results": []}


async def eval_intent(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    """意图识别评测：准确率 / Macro-F1。"""
    evaluator = _evaluator(config)
    cases = state.get("intent_cases")
    if not cases:
        return {}

    metrics = await evaluator._intent_evaluator.evaluate(cases)
    passed = metrics["accuracy"] >= evaluator.PASS_THRESHOLD
    result = EvalResult(
        test_id="intent_recognition",
        passed=passed,
        scores={"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
        detail=f"准确率 {metrics['accuracy']:.1%}，Macro-F1 {metrics['macro_f1']:.3f}",
        metadata={
            "total": metrics.get("total", 0),
            "correct": metrics.get("correct", 0),
            "cases": metrics.get("cases", []),
        },
    )
    return {"intent_metrics": metrics, "intent_result": [result]}


async def eval_dialog(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    """对话质量评测：调用 orchestrator 产出回复，再用 LLM Judge 逐轮评分。"""
    evaluator = _evaluator(config)
    cases = state.get("dialog_cases")
    if not cases:
        return {}

    results: List[EvalResult] = []
    for i, case in enumerate(cases):
        results.extend(await evaluator._evaluate_dialog_case(case, i))
    return {"dialog_results": results}


async def aggregate_scores(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    """汇总四维均分与通过率；四维只统计对话评分，意图准确率另以 intent_accuracy 计入。"""
    dialog_results = state["dialog_results"]

    all_scores: Dict[str, List[float]] = {
        "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
    }
    for r in dialog_results:
        for k in all_scores:
            if k in r.scores:
                all_scores[k].append(r.scores[k])

    avg_scores = {
        k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
    }
    intent_metrics = state["intent_metrics"]
    if intent_metrics:
        avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

    results = state["intent_result"] + dialog_results
    return {"results": results, "avg_scores": avg_scores}


async def detect_regression(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    evaluator = _evaluator(config)
    return {"regressions": evaluator._detect_regressions(state["avg_scores"])}


async def build_recommendations(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    evaluator = _evaluator(config)
    return {
        "recommendations": evaluator._recommendations(state["avg_scores"], state["intent_metrics"])
    }


async def build_report(state: EvalState, config: RunnableConfig) -> Dict[str, Any]:
    """生成报告并落基线；历史必须在回归比对之后才追加。"""
    evaluator = _evaluator(config)
    results = state["results"]
    passed_count = sum(1 for r in results if r.passed)
    pass_rate    = passed_count / len(results) if results else 0.0

    report = EvalReport(
        timestamp=datetime.now().isoformat(),
        total=len(results),
        passed=passed_count,
        pass_rate=round(pass_rate, 4),
        avg_scores=state["avg_scores"],
        regressions=state["regressions"],
        recommendations=state["recommendations"],
        results=results,
    )
    evaluator._history.append(report)
    evaluator._save_baseline(report)
    return {"report": report}


def build_eval_graph():
    graph = StateGraph(EvalState)
    graph.add_node("collect_cases", collect_cases)
    graph.add_node("intent_eval", eval_intent)
    graph.add_node("dialog_eval", eval_dialog)
    graph.add_node("aggregate", aggregate_scores)
    graph.add_node("regression", detect_regression)
    graph.add_node("recommendations", build_recommendations)
    graph.add_node("report", build_report)

    graph.add_edge(START, "collect_cases")
    # 两条分支总是各自执行一次（无对应用例时节点返回空更新），aggregate 才能静态等齐
    graph.add_edge("collect_cases", "intent_eval")
    graph.add_edge("collect_cases", "dialog_eval")
    graph.add_edge(["intent_eval", "dialog_eval"], "aggregate")
    graph.add_edge("aggregate", "regression")
    graph.add_edge("regression", "recommendations")
    graph.add_edge("recommendations", "report")
    graph.add_edge("report", END)
    return graph.compile()


EVAL_GRAPH = build_eval_graph()


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("我的订单什么时候到？",       "logistics"),
    IntentTestCase("我的订单还没发货",            "order_status"),
    IntentTestCase("你们服务太差了！",            "complaint"),
    IntentTestCase("应用一直报500错误",           "technical_crash"),
    IntentTestCase("为什么扣了两次款？",          "payment_issue"),
    IntentTestCase("我要投诉，转人工！",          "human_handoff"),
    IntentTestCase("你好",                        "greeting"),
    IntentTestCase("帮我开发票",                  "invoice"),
    IntentTestCase("退款多久到账？",              "refund"),
    IntentTestCase("登录一直报401",               "technical_login"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "我的订单 #12345 还没到，已经超时了"},
    {"question": "应用登录一直报错 401"},
    {"question": "为什么这个月多扣了 50 块钱？"},
    {"question": "我的退款什么时候能到账？"},
    {"turns": ["你好，我想退款", "订单号是 #12345", "退款多久能到账？"]},
]
