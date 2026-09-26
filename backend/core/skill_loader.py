"""
OptiServe Skill 加载器。

Skill 是一份可热加载的业务规范，走渐进式披露：system prompt 里只常驻它的
name + description 索引，正文由 Agent 自行判断后调用 load_skill 工具取回。
适合放置企业话术、处理流程、合规边界、排障 SOP 等需要运营侧快速调整的规则。
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 索引里单条 description 的展示上限；正文截断另有 max_body_chars 预算。
INDEX_DESCRIPTION_CHARS = 120

INDEX_PREAMBLE = (
    "以下 Skills 仅给出索引，正文未注入。\n"
    "判断需要哪一项后调用工具 load_skill(name) 取回全文并遵循；"
    "未取回全文前不要复述业务规则，也不要承诺时效、退款、赔偿或到货时间。"
)

INDEX_POSTSCRIPT = (
    "命中项建议优先加载；未命中但确属该业务场景时同样可以加载；不需要规范时直接回答。"
)


def _sanitize(value: str) -> str:
    """丢掉无法用 UTF-8 表达的字符，避免下游 json.dumps 抛孤立代理对。"""
    return value.encode("utf-8", errors="ignore").decode("utf-8")


def _normalize_name(value: str) -> str:
    # 连字符和下划线一并抹掉：Skill 名按官方规范是 kebab-case slug，
    # 模型回写成 technical_support 这类变体时仍要能命中。
    return value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


@dataclass
class Skill:
    """单个 Skill 的标准化表示，屏蔽 Markdown/JSON 等不同文件格式差异。"""
    name: str
    description: str
    content: str
    path: str
    dir_name: str = ""
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    enabled: bool = True

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
        """关键词命中结果，只用于提示与排序，永不决定可见性。"""
        lowered = (message or "").lower()
        return [keyword for keyword in self.keywords if keyword.lower() in lowered]

    def index_line(self, keywords_hit: List[str]) -> str:
        """索引里的一行；名称是 load_skill 的入参，必须逐字给出。"""
        description = self.description.strip()
        if len(description) > INDEX_DESCRIPTION_CHARS:
            description = description[:INDEX_DESCRIPTION_CHARS].rstrip() + "…"
        hint_text = f"【命中:{','.join(keywords_hit)}】" if keywords_hit else ""
        return f"- {self.name}：{description}{hint_text}"

    def body(self, max_chars: int) -> Tuple[str, bool]:
        """返回 (正文, 是否被截断)。"""
        text = self.content.strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "\n...", True
        return text, False

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
        }


class SkillManager:
    """
    从目录中发现、解析并管理 Skills。

    支持两种常用结构：
      1. skills/refund/SKILL.md（目录内的其它文件属于这个 Skill，不另立条目）
      2. skills/refund.json / skills/refund.md / skills/refund.txt
    """

    SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

    def __init__(
        self,
        root_dir: str,
        max_body_chars: int = 6000,
        max_index_chars: int = 1500,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_body_chars = max_body_chars
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
                skill = self._load_file(path)
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
        """发现可加载文件，优先读取目录规范文件 SKILL.md。

        技能目录内的其它文件属于那个 Skill，不另立条目——否则运营在目录里
        放一个 notes.md 就会凭空多出一条索引，还会被所有角色看到。
        """
        skill_md_files = sorted(root_dir.rglob("SKILL.md"))
        owned_dirs = {path.parent.resolve() for path in skill_md_files}
        for path in skill_md_files:
            yield path

        for path in sorted(root_dir.rglob("*")):
            if not path.is_file() or path.parent.resolve() in owned_dirs:
                continue
            if path.name.startswith(".") or path.name.upper() == "README.MD":
                continue
            if path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                yield path

    def _load_file(self, path: Path) -> Optional[Skill]:
        if path.suffix.lower() == ".json":
            return self._load_json(path)
        return self._load_text(path)

    def _load_json(self, path: Path) -> Optional[Skill]:
        raw = json.loads(_sanitize(path.read_text(encoding="utf-8")))
        if not isinstance(raw, dict):
            raise ValueError("JSON Skill 必须是对象格式")

        content = str(raw.get("content") or raw.get("instructions") or "").strip()
        if not content:
            raise ValueError("缺少 content 或 instructions")

        return Skill(
            name=str(raw.get("name") or path.stem),
            description=str(raw.get("description") or ""),
            content=content,
            path=str(path),
            dir_name=path.parent.name if path.name == "SKILL.md" else "",
            keywords=self._as_list(raw.get("keywords")),
            agents=[item.lower() for item in self._as_list(raw.get("agents"))],
            enabled=self._as_bool(raw.get("enabled")),
        )

    def _load_text(self, path: Path) -> Optional[Skill]:
        raw = _sanitize(path.read_text(encoding="utf-8"))
        meta, body = self._split_front_matter(raw)
        body = body.strip()
        if not body:
            return None

        is_dir_skill = path.name == "SKILL.md"
        default_name = path.parent.name if is_dir_skill else path.stem
        name = str(meta.get("name") or self._first_heading(body) or default_name)

        # 如果首行标题只是 Skill 名称，正文里就去掉它，减少重复噪音。
        body = self._strip_first_heading(body, name)

        return Skill(
            name=name,
            description=str(meta.get("description") or ""),
            content=body,
            path=str(path),
            dir_name=path.parent.name if is_dir_skill else "",
            keywords=self._as_list(meta.get("keywords")),
            agents=[item.lower() for item in self._as_list(meta.get("agents"))],
            enabled=self._as_bool(meta.get("enabled")),
        )

    def _split_front_matter(self, raw: str) -> Tuple[Dict[str, Any], str]:
        """
        解析 Markdown 顶部的简单 front matter。

        这里刻意不用 PyYAML，避免为一个轻量配置格式新增运行时依赖。
        列表支持两种写法：`key: v1, v2` 行内，和缩进的 `- v1` 块式。
        """
        text = raw.lstrip()
        if not text.startswith("---"):
            return {}, raw

        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw

        meta: Dict[str, Any] = {}
        last_key: Optional[str] = None
        end_idx: Optional[int] = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
            stripped = line.strip()
            if stripped.startswith("- ") and last_key is not None:
                current = meta.get(last_key)
                if not isinstance(current, list):
                    # 已写成行内值时不混用两种写法，避免静默覆盖。
                    if str(current or "").strip():
                        continue
                    current = []
                    meta[last_key] = current
                current.append(stripped[2:].strip().strip("\"'"))
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            meta[key] = value.strip().strip("\"'")
            last_key = key

        if end_idx is None:
            return {}, raw
        return meta, "\n".join(lines[end_idx + 1:])

    @staticmethod
    def _first_heading(body: str) -> Optional[str]:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
        return None

    @staticmethod
    def _strip_first_heading(body: str, name: str) -> str:
        lines = body.splitlines()
        if not lines:
            return body
        first = lines[0].strip()
        if first.startswith("#") and first.lstrip("#").strip() == name:
            return "\n".join(lines[1:]).strip()
        return body

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            items: Iterable[Any] = value
        else:
            # YAML 流式写法 [a, b] 和中文逗号都归一掉，否则元素会带着括号
            # 或被当成一整个关键词而永远匹配不上。
            items = str(value).strip().strip("[]{}").replace("，", ",").split(",")
        return [
            cleaned
            for cleaned in (
                str(item).strip().strip("\"'") for item in items
            ) if cleaned
        ]

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if value is None or value == "":
            return True
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
