"""转人工接手的演示操作，不创建工单、不连接任何外部系统。

注意：OptiServe 不会运行本文件。`run_skill_script` 只按文件名匹配到已登记的
演示操作，打一条 [skill-sim] 日志后回受理回执。本文件存在的意义是让
`skills/general-support` 的第三层清单非空，以及说明这类操作真实落地时
该长什么样。
"""
import logging

logger = logging.getLogger("optiserve.skills.create_human_handoff")


def main() -> int:
    logger.info("[demo] 转人工：真实实现应在此建单并返回排队位置与工单号")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
