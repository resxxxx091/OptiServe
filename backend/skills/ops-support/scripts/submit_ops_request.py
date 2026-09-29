"""演示脚本：登记一次商品/活动后台操作的人工受理。

本脚本从不被执行——run_skill_script 只登记调用并返回受理回执。
真实动作（改价、上下架、创建活动）由运营在商家后台人工完成。
"""
import logging

logger = logging.getLogger("optiserve.skills.submit_ops_request")


def main(args: dict | None = None) -> dict:
    args = args or {}
    logger.info("商品/活动操作人工受理登记: %s", args)
    return {"accepted": True, "note": "演示脚本，仅登记不执行"}
