"""
OptiServe Skill 加载器。

Skill 是一份可热加载的业务规范，走 Agent Skills 规范的三层渐进式披露：
第一层 name + description 索引常驻 system prompt；
第二层正文由 Agent 调用load_skill 取回；
第三层正文里指向的附表与演示脚本再由 Agent 按需取回或发起。
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

# Agent Skills 规范（https://agentskills.io/specification）。
# name 必需、1-64 字符、kebab-case、必须等于父目录名；description 必需、1-1024 字符。
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NAME_MAX_CHARS = 64
DESCRIPTION_MAX_CHARS = 1024
# 资源与脚本的扩展名白名单。
RESOURCE_SUFFIXES = {".md", ".txt", ".json", ".csv", ".yaml", ".yml"}
SCRIPT_SUFFIXES = {".py", ".sh"}


INDEX_PREAMBLE = (
    "以下 Skills 仅给出索引，正文未注入。\n"
    "判断需要哪一项后调用工具 load_skill(name) 取回全文；"
    "正文里指向的附表用 load_skill_resource(name, path) 读取，"
    "需要人工或二线处理的操作用 run_skill_script(name, script) 发起；"
)

INDEX_POSTSCRIPT = (
    "命中项建议优先加载；未命中但确属该业务场景时同样可以加载；不需要规范时直接回答。"
)


def _sanitize(value: str) -> str:
    """丢掉无法用 UTF-8 表达的字符，避免下游 json.dumps 抛孤立代理对。"""
    return value.encode("utf-8", errors="ignore").decode("utf-8")


def _normalize_name(value: str) -> str:
    """把 Skill 名称归一化为小写、去掉空格和下划线，方便模糊匹配。"""
    return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _normalize_rel_path(value: str) -> str:
    """把 Skill 资源或脚本的相对路径归一化为正斜杠、去掉开头的 ./，方便模糊匹配。2"""
    return (value or "").strip().replace("\\", "/").removeprefix("./")


@dataclass
class Skill:
    """单个 Skill 的标准化表示，由 SkillManager.load() 解析目录后生成。"""
    name: str
    description: str
    content: str
    path: str
    dir_name: str = ""
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    enabled: bool = True
    resources: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)

    @property
    def skill_dir(self) -> Path:
        return Path(self.path).parent

    def in_scope(self, agent_type: Optional[str]) -> bool:
        """可见性闸门：决定这个 Skill 出现在哪个 Agent 的索引里。

        - agents 为空：适用所有 Agent。
        - agent_type 缺失且 agents 非空：不列出。跨角色泄漏比漏列更贵。
        """
        if not self.enabled:
            return False
        if not self.agents:
            return True
        return bool(agent_type) and agent_type.lower() in self.agents

    def hint(self, message: str) -> List[str]:
        """关键词命中结果，只用于提示与排序，不决定可见性。"""
        lowered = (message or "").lower()
        return [keyword for keyword in self.keywords if keyword.lower() in lowered]

    def index_line(self, keywords_hit: List[str]) -> str:
        """索引里的一行；名称是 load_skill 的入参，必须逐字给出。"""
        description = self.description.strip()
        hint_text = f"【命中:{','.join(keywords_hit)}】" if keywords_hit else ""
        return f"- {self.name}：{description}{hint_text}"

    def to_summary(self) -> Dict[str, Any]:
        """返回 API 可序列化摘要，避免把完整长文本默认暴露给健康检查。"""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "keywords": self.keywords,
            "agents": self.agents,
            "enabled": self.enabled,
            "content_chars": len(self.content),
            "resources": self.resources,
            "scripts": self.scripts,
        }


class SkillManager:
    """
    从目录中发现、解析并管理 Skills。

    唯一结构：skills/<skill-name>/SKILL.md，目录内 references/ 与 scripts/属于这条 Skill 的第三层材料，不另立索引条目。
    """

    def __init__(
        self,
        root_dir: str,
        max_index_chars: int = 1500,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_index_chars = max_index_chars
        self._skills: List[Skill] = []
        self._errors: List[str] = []

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def load(self) -> List[Skill]:
        """重新扫描目录并加载 Skills；单个文件失败不会影响其他 Skill 生效。"""
        loaded: List[Skill] = []
        errors: List[str] = []

        if not self.root_dir.exists():
            logger.warning(f"Skill 目录不存在，跳过加载: {self.root_dir}")
            self._skills = []
            self._errors = []
            return []

        for path in self._discover_files(self.root_dir):
            try:
                skill = self._load_skill(path)
                if skill is not None:
                    loaded.append(skill)
            except Exception as ex:
                msg = f"{path}: {ex}"
                errors.append(msg)
                logger.warning(f"Skill 加载失败: {msg}")

        self._skills = loaded
        self._errors = errors
        self._log_loaded_skills()
        return self.skills

    def reload(self) -> List[Skill]:
        """运行时热加载入口，供 API 调用。"""
        return self.load()

    def catalog_for(self, agent_type: Optional[str] = None) -> List[Skill]:
        """该 Agent 可见的 Skills，顺序与加载顺序一致。"""
        return [skill for skill in self._skills if skill.in_scope(agent_type)]

    def hinted_for(self, message: str, agent_type: Optional[str] = None) -> List[Skill]:
        """关键词命中的可见 Skills，供「命中但未加载」探针使用。"""
        return [
            skill for skill in self.catalog_for(agent_type)
            if skill.hint(message)
        ]

    def index_for(self, message: str, agent_type: Optional[str] = None) -> str:
        """构建常驻 system prompt 插槽的 Skill 索引（只有 name + description）。"""
        ranked = sorted(
            ((skill, skill.hint(message)) for skill in self.catalog_for(agent_type)),
            key=lambda item: 0 if item[1] else 1,
        )
        if not ranked:
            logger.debug(
                "Skills 无可见条目: agent=%s message=%r",
                agent_type or "all",
                (message or "")[:80],
            )
            return ""

        remaining = self.max_index_chars - len(INDEX_PREAMBLE) - len(INDEX_POSTSCRIPT) - 2
        lines: List[str] = []
        dropped = 0
        for skill, keywords_hit in ranked:
            line = skill.index_line(keywords_hit)
            if remaining < len(line) + 1 and not keywords_hit:
                dropped += 1
                continue
            remaining -= len(line) + 1
            lines.append(line)

        if dropped:
            lines.append(f"（另有 {dropped} 项未列出，可直接用 load_skill(name) 加载）")

        logger.info(
            "Skills 已索引: agent=%s listed=%d hinted=%s message=%r",
            agent_type or "all",
            len(lines),
            ", ".join(skill.name for skill, hits in ranked if hits) or "none",
            (message or "")[:80],
        )
        return "\n".join((INDEX_PREAMBLE, *lines, INDEX_POSTSCRIPT))

    def body_for(self, name: str) -> Optional[Skill]:
        """按名称取回 Skill；禁用的 Skill 一律查不到，不向调用方披露开关。"""
        wanted = (name or "").strip()
        if not wanted:
            return None
        enabled = [skill for skill in self._skills if skill.enabled]
        for skill in enabled:
            if skill.name == wanted:
                return skill
        normalized = _normalize_name(wanted)
        for skill in enabled:
            if normalized and normalized in (
                _normalize_name(skill.name),
                _normalize_name(skill.dir_name),
            ):
                return skill
        return None

    def resource_for(self, skill: Skill, rel_path: str) -> Optional[str]:
        """第三层取文：只认 load() 时扫出的清单，模型给的变体一律不纠正。

        白名单来自启动/热加载时的目录扫描，因此扩展名与越界都被顺带挡住；
        清单外新建的文件要等一次 /skills/reload，与正文的热加载语义一致。
        """
        rel = _normalize_rel_path(rel_path)
        if rel not in skill.resources:
            return None
        root = skill.skill_dir.resolve()
        target = (skill.skill_dir / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        if not target.is_file():
            return None
        return _sanitize(target.read_text(encoding="utf-8"))

    @staticmethod
    def script_key(skill: Skill, rel_path: str) -> Optional[str]:
        """把模型回写的脚本路径归一到 scripts 清单条目上；命中不了返回 None。"""
        rel = _normalize_rel_path(rel_path)
        return rel if rel in skill.scripts else None

    def summary(self) -> Dict[str, Any]:
        """返回 Skill 管理器状态，用于 /skills 接口和排障。"""
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [skill.to_summary() for skill in self._skills],
            "errors": self.errors,
        }

    def _log_loaded_skills(self) -> None:
        """在控制台输出醒目的 Skill 加载结果，方便启动和热加载时确认生效状态。"""
        lines = [
            "",
            "================ OptiServe Skills Loaded ================",
            f"目录: {self.root_dir}",
            f"数量: {len(self._skills)}",
        ]

        if self._skills:
            for index, skill in enumerate(self._skills, start=1):
                agents = ", ".join(skill.agents) if skill.agents else "all"
                keywords = ", ".join(skill.keywords[:8]) if skill.keywords else "none"
                if len(skill.keywords) > 8:
                    keywords += ", ..."
                lines.extend([
                    f"{index}. {skill.name}",
                    f"   agents: {agents}",
                    f"   keywords: {keywords}",
                    f"   body_chars: {len(skill.content)}",
                    f"   resources: {', '.join(skill.resources) or 'none'}",
                    f"   scripts: {', '.join(skill.scripts) or 'none'}",
                    f"   path: {skill.path}",
                ])
        else:
            lines.append("未加载任何 Skill：Agent 的 [可用 Skills] 插槽将为空。")

        if self._errors:
            lines.append("解析错误:")
            lines.extend(f"  - {error}" for error in self._errors)

        lines.append("========================================================")
        message = "\n".join(lines)
        if self._skills:
            logger.info(message)
        else:
            logger.warning(message)

    def _discover_files(self, root_dir: Path) -> Iterable[Path]:
        """发现可加载文件：只有各 Skill 目录下的 SKILL.md 算一条 Skill。

        目录内的其它文件属于那个 Skill，不另立条目——否则运营在目录里
        放一个 notes.md 就会凭空多出一条索引，还会被所有角色看到。
        """
        yield from sorted(root_dir.rglob(SKILL_FILENAME))

    def _load_skill(self, path: Path) -> Optional[Skill]:
        raw = _sanitize(path.read_text(encoding="utf-8"))
        meta, body = _split_front_matter(raw)
        if not isinstance(meta, dict):
            raise ValueError("front matter 必须是键值映射")
        body = body.strip()

        name = str(meta.get("name") or "").strip()
        if not name:
            raise ValueError("front matter 缺少 name")
        if name != path.parent.name:
            raise ValueError(f"name={name!r} 必须与目录名 {path.parent.name!r} 逐字一致")
        if len(name) > NAME_MAX_CHARS or not SKILL_NAME_RE.match(name):
            raise ValueError(
                f"name={name!r} 不符合规范：1-{NAME_MAX_CHARS} 位小写字母/数字/单连字符，不以连字符开头结尾"
            )
        description = str(meta.get("description") or "").strip()
        if not description:
            raise ValueError("front matter 缺少 description")
        if len(description) > DESCRIPTION_MAX_CHARS:
            raise ValueError(f"description 超过 {DESCRIPTION_MAX_CHARS} 字")
        if not body:
            return None

        extra = meta.get("metadata") or {}
        if not isinstance(extra, dict):
            raise ValueError("metadata 必须是键值映射")
        resources, scripts = _scan_manifest(path.parent)

        return Skill(
            name=name,
            description=description,
            content=body,
            path=str(path),
            dir_name=path.parent.name,
            keywords=_as_list(extra.get("keywords")),
            agents=[item.lower() for item in _as_list(extra.get("agents"))],
            enabled=_as_bool(extra.get("enabled")),
            resources=resources,
            scripts=scripts,
        )


def _scan_manifest(skill_dir: Path) -> Tuple[List[str], List[str]]:
    """扫出第三层清单，只记相对路径；随 load() 刷新，不给每次请求重复走盘。"""
    root = skill_dir.resolve()
    resources: List[str] = []
    scripts: List[str] = []
    for entry in sorted(skill_dir.rglob("*")):
        if not entry.is_file() or entry.name == SKILL_FILENAME:
            continue
        rel = entry.relative_to(skill_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        try:
            entry.resolve().relative_to(root)
        except ValueError:
            continue
        rel_path = rel.as_posix()
        suffix = entry.suffix.lower()
        if "scripts" in rel.parts:
            if suffix in SCRIPT_SUFFIXES:
                scripts.append(rel_path)
        elif suffix in RESOURCE_SUFFIXES:
            resources.append(rel_path)
    return resources, scripts


def _split_front_matter(raw: str) -> Tuple[Any, str]:
    """切出 --- 包裹的 YAML front matter 与其后的正文。"""
    lines = raw.lstrip().splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            block = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1:])
            try:
                return yaml.safe_load(block) or {}, body
            except yaml.YAMLError as ex:
                raise ValueError(f"front matter 不是合法 YAML: {ex}") from None
    return {}, raw


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        items: Iterable[Any] = value
    else:
        # 逗号分隔的行内写法与中文逗号都归一掉，否则元素会带着括号
        # 或被当成一整个关键词而永远匹配不上。
        items = str(value).strip().strip("[]{}").replace("，", ",").split(",")
    return [
        cleaned
        for cleaned in (
            str(item).strip().strip("\"'") for item in items
        ) if cleaned
    ]


def _as_bool(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
