"""提交财务/人工审核：演示用操作，不连接工单系统、不写任何外部状态。

注意：OptiServe 不会运行本文件。`run_skill_script` 只按文件名匹配到已登记的
演示操作，打一条 [skill-sim] 日志后回受理回执。本文件存在的意义是让
`skills/billing-support` 的第三层清单非空，以及说明这类操作真实落地时
该长什么样。
"""
import logging

logger = logging.getLogger("optiserve.skills.submit_financial_review")


def main() -> int:
    logger.info("[demo] 提交财务审核：真实实现应在此建单并返回工单号")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
