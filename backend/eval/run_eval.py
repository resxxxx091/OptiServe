"""Langfuse 评测跑批：dataset 同步 → /chat 跑批 → 意图指标 + LLM-as-Judge → Score 挂载 → dataset run 关联。

产出（对应简历三条声称）：
  - 意图准确率 / 宏平均 F1（本地计算，gold 来自 dataset.json 的 intent 字段）
  - 端到端回答质量四维分（相关性/准确性/有用性/完整性），judge 用独立的 LLM 调用，
    以 NUMERIC Score 挂在每条评测 trace 上，理由写进 comment
  - 每条 trace 关联进同一次 Langfuse dataset run，UI 里可整批对比

用法（工作目录 backend/，.env 在仓库根目录）：
  PYTHONUTF8=1 uv run python eval/run_eval.py --sync-only     # 只同步数据集上 Langfuse
  PYTHONUTF8=1 uv run python eval/run_eval.py --limit 5       # 冒烟：只跑 5 条
  PYTHONUTF8=1 uv run python eval/run_eval.py                 # 全量
  PYTHONUTF8=1 uv run python eval/run_eval.py --no-judge      # 只出意图指标，不花 judge 的 token

前提：后端已启动（默认 127.0.0.1:8000）；.env 里 LANGFUSE_* 与 DEEPSEEK_API_KEY 有效。
缺 Langfuse 密钥时自动退化为纯本地模式：只算指标、不出 Score、不建 dataset run。
"""
import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import statistics
import sys
from collections import Counter
from typing import Dict
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

DATASET_NAME = "optiserve-eval"
RUN_PREFIX = "eval-run"
CHAT_TIMEOUT_S = 150.0
CONCURRENCY = 4
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "deepseek-chat")

JUDGE_PROMPT = """你是电商客服回答的质量评审员。对照参考要点，从四个维度给下面这条客服回答打分（0 到 1，一位小数）：
- relevance 相关性：回答是否针对用户的问题，有没有答非所问
- accuracy 准确性：内容是否与参考要点一致，有没有编造政策或事实
- helpfulness 有用性：用户拿到这个回答能不能继续行动，有没有实际帮助
- completeness 完整性：参考要点覆盖了多少，关键信息有没有遗漏

用户问题：
{question}

客服回答：
{answer}

参考要点（评 completeness 和 accuracy 的依据，不是逐字标准答案）：
{points}

只输出 JSON，不要其他文字：
{{"relevance": 0.0, "accuracy": 0.0, "helpfulness": 0.0, "completeness": 0.0, "reason": "<一句话说明>"}}"""


def _item_id(question: str) -> str:
    """数据集条目用确定性 id：重复同步是 upsert，不会越传越多。"""
    return f"optiserve-eval-{hashlib.md5(question.encode()).hexdigest()[:16]}"


def _auth_headers() -> Dict[str, str]:
    """后端鉴权中间件对所有路由生效（含 /search /chat），脚本必须带令牌。"""
    token = os.getenv("OPTISERVE_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _load_dataset(path: str) -> list:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    items = data["items"]
    if not items:
        raise SystemExit("dataset.json 没有条目")
    return items


def _sync_dataset(lf, items: list) -> None:
    """建 dataset 并按确定性 id upsert 全部条目；重复执行幂等。"""
    lf.create_dataset(
        name=DATASET_NAME,
        description="电商客服多 Agent 评测集（意图标注 + judge 参考要点）",
    )
    for item in items:
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=_item_id(item["question"]),
            input={"question": item["question"]},
            expected_output={"points": item["points"]},
            metadata={"intent": item["intent"]},
        )
    print(f"dataset '{DATASET_NAME}' 已同步 {len(items)} 条（upsert）")


async def _chat(client: httpx.AsyncClient, backend: str, question: str, semaphore: asyncio.Semaphore) -> dict:
    payload = {"message": question, "user_id": "eval"}
    async with semaphore:
        resp = await client.post(f"{backend}/chat", json=payload, timeout=CHAT_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


async def _judge(question: str, answer: str, points: list) -> dict:
    """独立的 judge LLM 调用：不走后端、不带 trace 上下文，避免污染被测链路的统计。"""
    from core.llm import LLMProvider, message_text
    from langchain_core.messages import HumanMessage

    chat = LLMProvider(
        os.getenv("DEEPSEEK_API_KEY", ""), os.getenv("DEEPSEEK_BASE_URL") or None
    ).chat_model(model=JUDGE_MODEL, temperature=0.0, max_tokens=512)
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, points="\n".join(f"- {p}" for p in points))
    resp = await chat.ainvoke([HumanMessage(content=prompt)])
    raw = message_text(resp)
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s < 0 or e <= s:
        raise ValueError(f"judge 输出没有 JSON: {raw[:120]}")
    scores = json.loads(raw[s:e])
    return {
        "relevance": float(scores["relevance"]),
        "accuracy": float(scores["accuracy"]),
        "helpfulness": float(scores["helpfulness"]),
        "completeness": float(scores["completeness"]),
        "reason": str(scores.get("reason", "")),
    }


def _intent_metrics(results: list) -> dict:
    """准确率 + 宏平均 F1（对标注中出现的真实意图求宏平均，other 兜底类不计入宏平均）。"""
    golds = [r["intent_gold"] for r in results]
    preds = [r["intent_pred"] for r in results]
    accuracy = sum(g == p for g, p in zip(golds, preds)) / len(results)
    labels = sorted({g for g in golds} - {"other"})
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "labels": len(labels),
        "confusions": [
            {"gold": g, "pred": p, "count": c}
            for (g, p), c in Counter(zip(golds, preds)).items() if g != p
        ],
    }


def _mount_scores(lf, trace_id: str, item: dict, result: dict) -> None:
    """四维 judge 分 + 意图对错挂到 trace；judge 失败只挂意图分。"""
    judge = result.get("judge")
    if judge:
        for dim in ("relevance", "accuracy", "helpfulness", "completeness"):
            lf.create_score(
                trace_id=trace_id, name=dim, value=judge[dim],
                data_type="NUMERIC", comment=judge.get("reason", ""),
            )
    lf.create_score(
        trace_id=trace_id, name="intent_correct",
        value=1.0 if result["intent_gold"] == result["intent_pred"] else 0.0,
        data_type="BOOLEAN",
    )


def _link_dataset_run(lf, run_name: str, item: dict, trace_id: str, result: dict) -> None:
    from langfuse.api import CreateDatasetRunItemRequest

    lf.api.dataset_run_items.create(
        request=CreateDatasetRunItemRequest(
            runName=run_name,
            datasetItemId=_item_id(item["question"]),
            traceId=trace_id,
            metadata={
                "intent_gold": result["intent_gold"],
                "intent_pred": result["intent_pred"],
                "intent_correct": result["intent_gold"] == result["intent_pred"],
            },
        ).dict(exclude_none=True)
    )


def _langfuse_client():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("未配置 LANGFUSE_* 密钥：退化为纯本地模式（只有意图指标，无 Score / dataset run）")
        return None
    from langfuse import Langfuse

    return Langfuse()


async def run_rag(args) -> None:
    """RAG 检索评测：rag_dataset.json 的问题打 /search，按标题命中 rag_expect 算指标。

    /search 的 trace_id 是纯随机 uuid，脚本侧无法复算，所以 RAG 指标只落本地报告，
    不像意图评测那样把 Score 挂回 Langfuse trace。
    """
    path = pathlib.Path(__file__).parent / "rag_dataset.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_one(client: httpx.AsyncClient, item: dict) -> dict:
        expects = item["rag_expect"] if isinstance(item["rag_expect"], list) else [item["rag_expect"]]
        result = {"question": item["question"], "rag_expect": expects}
        try:
            params = httpx.QueryParams({"query": item["question"], "top_k": args.top_k})
            async with semaphore:
                resp = await client.post(f"{args.backend}/search?{params}", timeout=TIMEOUT_SEARCH)
            resp.raise_for_status()
            data = resp.json()
        except Exception as ex:
            result.update(error=f"{type(ex).__name__}: {ex}")
            return result
        titles = [(r.get("title") or "") for r in (data.get("results") or [])]
        # 命中 = 返回文档标题包含标注的文档标题子串
        hits = [1 if any(key in title for title in titles) else 0 for key in expects]
        first_rank = next(
            (i + 1 for i, t in enumerate(titles) if any(key in t for key in expects)), 0
        )
        result.update(
            retrieved=titles,
            recall=sum(hits) / len(expects),
            precision=sum(hits) / args.top_k,
            mrr=1.0 / first_rank if first_rank else 0.0,
            degraded=data.get("degraded", False),
        )
        return result

    async with httpx.AsyncClient(headers=_auth_headers()) as http:
        results = await asyncio.gather(*(run_one(http, item) for item in items))

    ok = [r for r in results if "error" not in r]
    print(f"\nRAG 评测完成：成功 {len(ok)} / 失败 {len(results) - len(ok)}，top_k={args.top_k}")
    if ok:
        recall = statistics.mean(r["recall"] for r in ok)
        precision = statistics.mean(r["precision"] for r in ok)
        mrr = statistics.mean(r["mrr"] for r in ok)
        hit = sum(1 for r in ok if r["recall"] == 1.0) / len(ok)
        print(f"召回率 recall@{args.top_k} {recall:.1%}  精确率 precision@{args.top_k} {precision:.1%}")
        print(f"命中率（期望文档全部进 top-k） {hit:.1%}  MRR {mrr:.3f}")
        worst = Counter()
        for r in ok:
            if r["recall"] < 1.0:
                for key in r["rag_expect"]:
                    worst[key] += 1
        for key, count in worst.most_common(10):
            print(f"  召回薄弱文档：{key} ×{count}")

    report = pathlib.Path(__file__).parent / "rag_report.json"
    report.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细报告：{report}")


TIMEOUT_SEARCH = 90.0


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=os.getenv("EVAL_BACKEND", "http://127.0.0.1:8000"))
    parser.add_argument("--dataset", default=str(pathlib.Path(__file__).parent / "dataset.json"))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    parser.add_argument("--sync-only", action="store_true", help="只同步数据集到 Langfuse，不跑批")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-Judge，只出意图指标")
    parser.add_argument("--rag", action="store_true", help="RAG 检索评测：跑 /search 算 recall/precision/MRR")
    parser.add_argument("--top-k", type=int, default=5, help="RAG 评测的截断窗口 k")
    args = parser.parse_args()

    if args.rag:
        await run_rag(args)
        return

    items = _load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    lf = _langfuse_client()
    if lf and not args.no_judge:
        _sync_dataset(lf, items)
    if args.sync_only:
        return

    run_name = f"{RUN_PREFIX}-{datetime.now():%Y%m%d-%H%M%S}"
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(headers=_auth_headers()) as http:
        async def run_one(item: dict) -> dict:
            result = {"question": item["question"], "intent_gold": item["intent"], "points": item["points"]}
            try:
                raw = await _chat(http, args.backend, item["question"], semaphore)
            except Exception as ex:
                result.update(error=f"{type(ex).__name__}: {ex}", intent_pred="", response="")
                return result
            result.update(
                request_id=raw.get("request_id", ""),
                response=raw.get("response", ""),
                intent_pred=raw.get("intent", ""),
                latency_ms=raw.get("latency_ms"),
            )
            if not args.no_judge:
                try:
                    result["judge"] = await _judge(item["question"], result["response"], item["points"])
                except Exception as ex:
                    result["judge_error"] = f"{type(ex).__name__}: {ex}"
            return result

        results = await asyncio.gather(*(run_one(item) for item in items))

    ok = [r for r in results if "error" not in r]
    failed = len(results) - len(ok)
    print(f"\n跑批完成：成功 {len(ok)} / 失败 {failed}")

    if ok:
        metrics = _intent_metrics(ok)
        print(f"意图准确率 {metrics['accuracy']:.1%}  宏平均 F1 {metrics['macro_f1']:.4f}（{metrics['labels']} 类，不含 other）")
        for c in metrics["confusions"]:
            print(f"  误判 {c['gold']} → {c['pred']} ×{c['count']}")
        if not args.no_judge:
            judged = [r for r in ok if r.get("judge")]
            for dim in ("relevance", "accuracy", "helpfulness", "completeness"):
                vals = [r["judge"][dim] for r in judged]
                print(f"judge {dim}: {statistics.mean(vals):.3f}（{len(vals)} 条）" if vals else f"judge {dim}: 无数据")

    if lf:
        for item, result in zip(items, results):
            if "error" in result:
                continue
            # trace_id 由 request_id seed 派生，与后端 tracing.py:109 同一约定，脚本侧可复算
            trace_id = lf.create_trace_id(seed=f"optiserve:{result['request_id']}")
            _mount_scores(lf, trace_id, item, result)
            _link_dataset_run(lf, run_name, item, trace_id, result)
        lf.flush()
        print(f"\nScore 已挂载，dataset run：{run_name}")

    report_path = pathlib.Path(__file__).parent / "eval_report.json"
    report_path.write_text(json.dumps(
        {"run_name": run_name, "metrics": _intent_metrics(ok) if ok else {}, "results": results},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"明细报告：{report_path}")


if __name__ == "__main__":
    asyncio.run(main())
