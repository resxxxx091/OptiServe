# OptiServe Skills 文档

OptiServe 启动时会从 `OPTISERVE_SKILLS_DIR` 读取 Skills。Skills 采用渐进式披露：只有 front matter 的 `name` + `description` 组成索引，常驻在对应 Agent 的 system prompt 插槽；规范正文不注入，由 Agent 判断需要后调用 `load_skill(name)` 工具按需取回。Skills 适合维护业务处理规范、客服话术、技术排障 SOP、账单审核边界、升级规则和禁止事项。

三段分工：`agents` 决定哪几篇进入该 Agent 的索引（可见性闸门），`keywords` 只决定索引里哪几条带 `【命中:…】` 标记并排在前面（提示，不闸门），是否加载正文由答题的 Agent 决定。

当前内置四类 Skills：

```text
skills/general-customer-service/SKILL.md  # 通用客服：接待、澄清、分流、投诉和转人工
skills/technical-support/SKILL.md         # 技术支持：故障排查、接口错误、部署配置和安全边界
skills/billing-support/SKILL.md           # 账单服务：扣款、退款、发票、订阅和财务审核
skills/order-support/SKILL.md             # 订单支持：订单状态、发货进度、配送时效和收货信息变更
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill-name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: technical-support
description: 故障排查、错误码解读、配置指导与升级条件的口径。用户报告报错、无法登录、接口或 SDK 异常、超时崩溃时加载本规范
keywords: 报错,错误,接口,API,部署,超时,500,401,日志
agents: technical
enabled: true
---
```

字段说明：

- `name`：Skill 名称，同时是 `load_skill` 的入参，按官方规范取 **kebab-case slug**——只允许小写字母、数字和连字符，不以连字符开头或结尾，不超过 64 字符，且**必须与所在目录名逐字一致**。中文标题写在正文首行的 `# 技术支持处理规范` 里，不进 `name`。模型必须逐字复制索引里的名称，`body_for()` 允许大小写、空白、下划线与连字符互换后的规范化匹配，以及按目录名匹配。
- `description`：唯一常驻上下文的字段，直接决定模型会不会去加载正文。用第三人称 + 触发场景写法（"…的口径与禁止事项。用户问到 X、Y、Z 时加载本规范"）；索引里超过 120 字会被截断。**不要写"你是…专家"这类角色自称**，测试的调用归属依赖各角色 system prompt 的开头短语。
- `keywords`：只产生 `【命中:…】` 标记与排序权重，**不决定是否加载、也不让该技能从索引里消失**；空 `keywords` 只代表永不提示。多个关键词用英文逗号 `,` 分隔，也支持缩进的 `- 关键词` 块式写法；中文逗号会被归一，但建议统一用英文逗号。
- `agents`：可见性闸门，可填 `general`、`technical`、`billing`、`order`，多个值用逗号分隔；留空表示适用于所有角色。
- `enabled`：是否启用，支持 `true/false`。`false` 的技能既不进索引，`load_skill` 也查不到。

## 编写要求

- 单篇正文超过 `OPTISERVE_SKILL_MAX_BODY_CHARS`（默认 6000）字会被截断，返回结果的 `truncated` 字段会标出。
- 一类 Skill 只描述一类职责，不要把技术、账单、通用客服规则混在一个文件里。
- 必须包含"角色定位""处理流程""升级条件""禁止事项"等稳定章节。
- 对用户隐私、支付、密码、验证码、API Key、Token 等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如"通常""预计""需要核验后确认"。
- 对需要人工、财务、二线技术处理的场景要明确写出升级条件。
- 一条 Skill 内的规则只有被加载后才生效，因此把"何时该用我"写进 `description`，而不是只写在正文里。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```

验证是否真的生效（`load_skill` 走晚绑定闭包，始终读当前 SkillManager）：

```bash
# 1) 索引条目与正文字数
curl -s http://localhost:8000/skills | jq '.skills[] | {name, description, content_chars}'
# 2) 发一次命中该 Skill 的对话，取 request_id
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"我要退款，什么时候到账","user_id":"u1"}' | jq '{request_id, tools_used}'
# 3) 用上一步的 request_id 看工具轨迹，tools_used 里应出现 load_skill
curl -s http://localhost:8000/trace/tool/<request_id> | jq '.trace | {tools_used, inputs: [.tool_calls[].input]}'
```

启动日志里出现 `未加载任何 Skill` 的 WARNING，通常意味着 `OPTISERVE_SKILLS_DIR` 指向了空目录或挂载路径不对。
