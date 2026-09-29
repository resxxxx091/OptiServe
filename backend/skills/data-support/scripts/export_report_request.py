"""演示脚本：登记一次真实业务库报表导出的人工受理。

本脚本从不被执行——run_skill_script 只登记调用并返回受理回执。
真实动作发生在线下：运营在后台用真实业务库导出，不经过本系统。
"""
import logging

logger = logging.getLogger("optiserve.skills.export_report_request")


def main(args: dict | None = None) -> dict:
    args = args or {}
    logger.info("报表导出人工受理登记: %s", args)
    return {"accepted": True, "note": "演示脚本，仅登记不执行"}
