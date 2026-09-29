"""演示脚本：登记一次人工升级受理，不创建工单、不连接任何外部系统。

注意：OptiServe 不会运行本文件。run_skill_script 只按文件名匹配到已登记的
演示操作，打一条 [skill-sim] 日志后回受理回执。
"""
import logging

logger = logging.getLogger("optiserve.skills.create_human_handoff")


def main(args: dict | None = None) -> dict:
    args = args or {}
    logger.info("人工升级受理登记: %s", args)
    return {"accepted": True, "note": "演示脚本，仅登记不执行"}
