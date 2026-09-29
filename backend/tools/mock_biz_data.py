"""模拟业务数据源：为数据查询/报表/异常诊断类意图提供确定性读数。

这是演示环境的假数据——用固定 seed 生成 30 天的店铺指标、8 个 SKU 的库存
和一批异常订单。三个特点：
  - 确定性：同一 (seed, 参数) 永远返回同一批数，评测和演示可复现；
  - 只读：所有工具都只查不改，任何"改价/下架"动作都必须走人工受理回执；
  - 脱敏：订单只有尾号，没有买家个人信息。

真实接入时把 _load() 换成业务库查询即可，工具契约不变。
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Dict, List

_SEED = 20240915
_TODAY = date(2024, 9, 15)  # 固定"今天"，保证数据不随真实时钟漂移
_DAYS = 30

PRODUCTS: List[Dict[str, Any]] = [
    {"sku": "SKU-1001", "name": "无线蓝牙耳机", "category": "数码", "price": 199.0, "stock": 326, "status": "在售"},
    {"sku": "SKU-1002", "name": "保温杯", "category": "家居", "price": 89.0, "stock": 88, "status": "在售"},
    {"sku": "SKU-1003", "name": "儿童书包", "category": "母婴", "price": 129.0, "stock": 45, "status": "在售"},
    {"sku": "SKU-1004", "name": "电饭煲", "category": "家电", "price": 329.0, "stock": 12, "status": "在售"},
    {"sku": "SKU-1005", "name": "防晒霜", "category": "美妆", "price": 79.0, "stock": 0, "status": "缺货"},
    {"sku": "SKU-1006", "name": "运动鞋", "category": "服饰", "price": 259.0, "stock": 154, "status": "在售"},
    {"sku": "SKU-1007", "name": "手机壳", "category": "数码", "price": 29.0, "stock": 902, "status": "在售"},
    {"sku": "SKU-1008", "name": "蓝牙音箱", "category": "数码", "price": 159.0, "stock": 67, "status": "下架中"},
]

ANOMALOUS_ORDERS: List[Dict[str, Any]] = [
    {"order_id": "SO-2409-8841", "tail": "8841", "sku": "SKU-1004", "issue": "买家申请退款，金额与实付不符", "age_hours": 30, "status": "待审核"},
    {"order_id": "SO-2409-8852", "tail": "8852", "sku": "SKU-1002", "issue": "地址重复修改两次，物流已揽收", "age_hours": 18, "status": "待处理"},
    {"order_id": "SO-2409-8867", "tail": "8867", "sku": "SKU-1001", "issue": "支付成功但订单未生成", "age_hours": 9, "status": "待核实"},
    {"order_id": "SO-2409-8870", "tail": "8870", "sku": "SKU-1006", "issue": "客诉升级：三天未发货", "age_hours": 52, "status": "待审核"},
    {"order_id": "SO-2409-8874", "tail": "8874", "sku": "SKU-1003", "issue": "疑似恶意退货（半年第7次）", "age_hours": 6, "status": "待核实"},
]

# 30 天店铺指标：工作日/周末用不同基线，再叠一点趋势与噪声，让"查数据/找异常"有真实感
def _daily_metrics() -> List[Dict[str, Any]]:
    rng = random.Random(_SEED)
    rows: List[Dict[str, Any]] = []
    for offset in range(_DAYS - 1, -1, -1):
        day = _TODAY - timedelta(days=offset)
        weekend_boost = 1.35 if day.weekday() >= 5 else 1.0
        # 9 月 8 日起转化率持续下滑，给"异常诊断"留一个可发现的真实模式
        trend = 1.0 if day < date(2024, 9, 8) else 1.0 - 0.03 * (day - date(2024, 9, 8)).days
        visitors = int(5200 * weekend_boost * rng.uniform(0.9, 1.1))
        conversion = max(0.005, 0.032 * trend * rng.uniform(0.92, 1.08))
        orders = int(visitors * conversion)
        aov = rng.uniform(88, 132)
        gmv = round(orders * aov, 2)
        refund_rate = round(rng.uniform(0.03, 0.07), 4)
        rows.append({
            "date": day.isoformat(),
            "visitors": visitors,
            "orders": orders,
            "gmv": gmv,
            "conversion": round(conversion, 4),
            "refund_rate": refund_rate,
        })
    return rows


METRICS: List[Dict[str, Any]] = _daily_metrics()


def query_metrics(req, args: Dict[str, Any]) -> Dict[str, Any]:
    """按日期范围汇总店铺核心指标；days 缺省 7。"""
    try:
        days = min(max(int(args.get("days", 7)), 1), _DAYS)
    except (TypeError, ValueError):
        return {"success": False, "error": "days 必须是整数"}
    window = METRICS[-days:]
    total_gmv = round(sum(r["gmv"] for r in window), 2)
    total_orders = sum(r["orders"] for r in window)
    total_visitors = sum(r["visitors"] for r in window)
    prev = METRICS[-2 * days:-days] if len(METRICS) >= 2 * days else []
    prev_gmv = round(sum(r["gmv"] for r in prev), 2) if prev else None
    return {
        "success": True,
        "range": {"days": days, "from": window[0]["date"], "to": window[-1]["date"]},
        "gmv": total_gmv,
        "orders": total_orders,
        "visitors": total_visitors,
        "conversion": round(total_orders / total_visitors, 4) if total_visitors else 0.0,
        "avg_refund_rate": round(sum(r["refund_rate"] for r in window) / len(window), 4),
        "gmv_wow_change": round((total_gmv - prev_gmv) / prev_gmv, 4) if prev_gmv else None,
        "daily": window,
        "disclaimer": "演示环境模拟数据，仅用于功能演示与评测",
    }


def query_inventory(req, args: Dict[str, Any]) -> Dict[str, Any]:
    """按 SKU 或商品名模糊查库存与状态。"""
    keyword = str(args.get("keyword", "")).strip()
    if not keyword:
        return {"success": False, "error": "keyword 不能为空", "available": [p["name"] for p in PRODUCTS]}
    hits = [p for p in PRODUCTS if keyword in p["name"] or keyword.upper() == p["sku"]]
    return {
        "success": True,
        "matches": hits,
        "low_stock_threshold": 50,
        "disclaimer": "演示环境模拟数据，仅用于功能演示与评测",
    }


def query_anomalous_orders(req, args: Dict[str, Any]) -> Dict[str, Any]:
    """列出当前待处理的异常订单；status 可过滤。"""
    status = str(args.get("status", "")).strip()
    rows = [o for o in ANOMALOUS_ORDERS if not status or o["status"] == status]
    return {
        "success": True,
        "count": len(rows),
        "orders": rows,
        "note": "处理动作需要人工确认，本工具只提供查询",
        "disclaimer": "演示环境模拟数据，仅用于功能演示与评测",
    }
