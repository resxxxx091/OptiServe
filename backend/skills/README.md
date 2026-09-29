# OptiServe Skills 文档

OptiServe 启动时会从 `OPTISERVE_SKILLS_DIR` 读取 Skills。Skills 采用三层渐进式披露：只有 front matter 的 `name` + `description` 组成索引，常驻在对应 Agent 的 system prompt 插槽；规范正文不注入，由 Agent 判断需要后调用 `load_skill(name)` 取回；正文再指向的附表与演示操作由 `load_skill_resource` / `run_skill_script` 按需取回。Skills 适合维护业务处理规范、数据分析口径、活动配置清单、订单客诉处理流程、系统故障排障 SOP、升级规则和禁止事项。

三段分工：`metadata.agents` 决定哪几篇进入该 Agent 的索引（可见性闸门），`metadata.keywords` 只决定索引里哪几条带 `【命中:…】` 标记并排在前面（提示，不闸门），是否加载正文由答题的 Agent 决定。

当前内置四类 Skills，四条都带第三层：

```text
skills/general-support/
├── SKILL.md
├── references/routing-map.md            # 现象→角色去向、转接前要总结的字段
└── scripts/create_human_handoff.py      # 转人工/专家受理

skills/data-support/
├── SKILL.md
├── references/metric-definitions.md     # 指标口径对照与常见误解
└── scripts/export_report_request.py     # 真实业务库导出的人工受理

skills/ops-support/
├── SKILL.md
├── references/campaign-checklist.md     # 活动类型×必备要素清单
└── scripts/submit_ops_request.py        # 改价/上下架/创建活动的人工受理

skills/service-support/
├── SKILL.md
├── references/order-issue-matrix.md     # 订单客诉场景×处理边界
└── scripts/submit_service_escalation.py # 工单升级的人工受理

四张附表都**不含时效与金额数值**：正文没有规定的数字，附表里也不能出现，否则模型加载完正文又会从附表里捡一个编造的时间点承诺给用户。需要具体时限一律转人工核实。

## Skill 目录结构

每个 Skill 是一个目录，遵循上游 Agent Skills 规范的三层渐进式披露：

```text
skills/<skill-name>/
├── SKILL.md          # 必需：front matter + 规范正文（第一、二层）
├── references/*.md   # 可选：附表与参考文档（第三层，按需读取）
└── scripts/*.py      # 可选：登记的演示操作（第三层，见下文）
```

只有 `SKILL.md` 会被解析成一条 Skill；`references/` 与 `scripts/` 里的文件挂在它名下，不另立索引条目。根目录下的散装 `.md`/`.txt` 和 JSON 文件一律不认。

三层各自的入口：`name` + `description` 组成常驻索引 → `load_skill(name)` 取回正文 → 正文指向的附表用 `load_skill_resource(name, path)` 读取、演示操作用 `run_skill_script(name, script)` 发起。`load_skill` 的返回体带着 `resources`/`scripts` 清单，那是模型唯一的路径来源。

第三层的取舍标准：**大且不每次都用**的东西才拆出去（错误码对照表、费率表、长话术模板）。几乎每轮都要引用的内容留在正文里，拆出去只会多一次模型往返。

front matter 是 `---` 包裹的 YAML：

```markdown
---
name: service-support
description: 订单履约与客诉处理规范：异常订单处理路径、退款审核边界、客诉升级条件、系统故障初步排查。运营问退款审核、异常订单、客诉升级、后台报错时加载本规范
metadata:
  keywords: 订单,退款,审核,异常订单,客诉,投诉,工单,升级,报错,500,登录,故障
  agents: service
  enabled: true
---
```

字段说明：

- `name`（顶层，必需）：同时是三个工具的 `name` 入参。取 **kebab-case slug**——只允许小写字母、数字和单个连字符，不以连字符开头或结尾，不超 64 字符，且**必须与所在目录名逐字一致**，否则该条不加载并出现在 `GET /skills` 的 `errors` 里。中文标题写在正文首行的 `# 技术支持处理规范` 里，不进 `name`。模型必须逐字复制索引里的名称，`body_for()` 允许大小写、空白、下划线与连字符互换后的规范化匹配，以及按目录名匹配。
- `description`（顶层，必需）：唯一常驻上下文的字段，直接决定模型会不会去加载正文。非空、不超 1024 字，索引里整条给出、不截断——写多长就常驻多长，所以只写"是什么 + 什么时候用"，细节进正文。用第三人称 + 触发场景写法；**不要写"你是…专家"这类角色自称**，测试的调用归属依赖各角色 system prompt 的开头短语。
- `license` / `compatibility`：规范里的可选顶层字段，本项目当前不解析，写了会被忽略。

自定义字段收在 `metadata:` 下（规范的 `metadata` 是任意键值映射，这样写能整份拷给其它支持该规范的客户端）：

- `metadata.keywords`：只产生 `【命中:…】` 标记与排序权重，**不决定是否加载、也不让该技能从索引里消失**；空值只代表永不提示。多个关键词用英文逗号分隔，中文逗号会被归一。
- `metadata.agents`：可见性闸门，可填 `general`、`data`、`ops`、`service`，逗号分隔；留空表示适用于所有角色。
- `metadata.enabled`：是否启用，支持 `true/false`。`false` 的技能既不进索引，三个工具也都查不到。

## scripts/ 是演示操作，不会被执行

`scripts/` 下的脚本只贡献清单和可读源码，**服务端从不运行它们**：`run_skill_script` 校验名字在清单内之后打一条 `[skill-sim]` 日志，返回 `{"status": "accepted", "detail": "已受理，等待人工核验"}`。所以模型相信自己走完了一次升级流程，而真实动作发生在线下。

由此两条约束：

- 回执只说"已受理"，不要指望它承诺结果——账单类规范本就规定缺少核验信息时不得承诺退款成功、到账时间，回执不能替模型破它自己的规矩。
- 脚本源码对模型不可见：`load_skill_resource` 只认 `resources` 清单，而清单扫描把 `.py`/`.sh` 只归到 `scripts/` 下、且那条路走不到执行。要让模型参考实现细节，就把说明写进 `references/`。
- 四条内置 Skill 的 `scripts/` 各登记一条升级/受理操作，正文里都单列一节 `## 附表与升级操作` 指路。新增一条操作 = 放一个脚本文件 + 在正文加一行引用 + 走一次 reload。

## 编写要求

- 一类 Skill 只描述一类职责，不要把数据、商品活动、订单客诉、通用接待的规则混在一个文件里。
- 正文建议控制在 500 行以内，超出就把附表拆到 `references/`；本项目对正文和附表都**不设字数上限**，体量完全靠编写者自觉。
- 引用第三层文件用相对 `SKILL.md` 的路径，只引一层，且要和清单里的字符串逐字一致（`load_skill_resource` / `run_skill_script` 只容忍 `./` 前缀与反斜杠两种变体，裸文件名会被拒）。
- 必须包含"角色定位""处理流程""升级条件""禁止事项"等稳定章节。
- 对用户隐私、支付、密码、验证码、API Key、Token 等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如"通常""预计""需要核验后确认"。
- 对需要人工核验、财务审核、二线技术处理的场景要明确写出升级条件。
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

验证是否真的生效（三个 Skill 工具都走晚绑定闭包，始终读当前 SkillManager）：

```bash
# 1) 索引条目、正文字数与第三层清单；name 与目录名不一致等校验失败会出现在 errors[] 里
curl -s http://localhost:8000/skills | jq '{errors, skills: [.skills[] | {name, content_chars, resources, scripts}]}'
# 2) 发一次命中该 Skill 的对话：响应体的 tools_used 里应出现 load_skill
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"上周的销售额是多少","user_id":"u1"}' | jq '{request_id, tools_used}'
# 3) 改了 references/ 下的附表同样先 reload，再发一条会用到码表的对话，
#    tools_used 里应出现 load_skill_resource；这一跳不需要重启进程
# 4) 想看这一跳的完整 span 树：拿 request_id 去 Langfuse 搜，本地不留链路记录
```

`load()` 会同时重扫第三层清单，所以新增或删除 `references/`、`scripts/` 里的文件也必须走一次 reload——模型能拿到的路径只有清单里那些。

启动日志里出现 `未加载任何 Skill` 的 WARNING，通常意味着 `OPTISERVE_SKILLS_DIR` 指向了空目录或挂载路径不对。
