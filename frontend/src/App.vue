<template>
  <main :class="['app-shell', `app-shell-${activeView}`]">
    <header class="topbar">
      <a class="brand" href="#" aria-label="OptiServe 首页" @click.prevent="activeView = 'chat'">
        <span class="brand-mark">OS</span>
        <span class="brand-name">OptiServe<small>调试工作台</small></span>
      </a>

      <nav class="view-nav" aria-label="工作区">
        <button :class="{ active: activeView === 'chat' }" @click="activeView = 'chat'">对话</button>
        <button :class="{ active: activeView === 'knowledge' }" @click="activeView = 'knowledge'">知识库</button>
        <button :class="{ active: activeView === 'trace' }" @click="openTraceView">链路</button>
        <button :class="{ active: activeView === 'evaluation' }" @click="activeView = 'evaluation'">评测</button>
      </nav>

      <div class="topbar-tools">
        <span class="environment-pill">
          <i :class="healthOk ? 'online' : 'offline'"></i>
          {{ healthLabel }}
        </span>
        <span class="topbar-divider" aria-hidden="true"></span>
        <a class="docs-link" :href="docsUrl" target="_blank" rel="noreferrer">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
            <path d="M6.5 3.5h-2A1.5 1.5 0 0 0 3 5v6a1.5 1.5 0 0 0 1.5 1.5h6A1.5 1.5 0 0 0 12 11V9" />
            <path d="M9.5 2.5H13.5V6.5M13.5 2.5 8 8" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          API 文档
        </a>
        <button class="avatar-button" title="当前用户">{{ userInitial }}</button>
      </div>
    </header>

    <div v-if="toast" class="toast" role="status">{{ toast }}</div>

    <section v-if="activeView === 'chat'" class="page page-chat">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">POST /chat</span>
          <h1>和客服 Agent 对话</h1>
          <p>发送一条真实请求，查看它如何识别意图、选择 Agent 并生成回复。</p>
        </div>
        <div class="heading-actions">
          <span class="session-label">conv: {{ settings.conversationId || 'new' }}</span>
          <button class="quiet-button" @click="clearConversation">清空</button>
        </div>
      </div>

      <div class="chat-layout">
        <section class="chat-stage">
          <div class="stage-bar">
            <div class="stage-context">
              <span class="context-dot"></span>
              <span>{{ API_BASE }}</span>
            </div>
            <span>{{ messages.length }} 条消息</span>
          </div>

          <div class="messages" ref="messageList">
            <article
              v-for="item in messages"
              :key="item.id"
              :class="['message', item.role, { degraded: item.degraded }]"
            >
              <div class="message-meta">
                <span>{{ item.role === 'user' ? 'user' : 'agent' }}</span>
                <small v-if="item.meta">{{ item.meta }}</small>
                <em v-if="item.degraded" class="degraded-tag">DEGRADED {{ item.degradations?.length }}</em>
              </div>
              <p>{{ item.content }}</p>
              <ul v-if="item.degradations?.length" class="degrade-list">
                <li v-for="(event, index) in item.degradations" :key="`${event.source}-${event.code}-${index}`">
                  <code>{{ event.source }}/{{ event.code }}</code>
                  <span>{{ event.message }}</span>
                </li>
              </ul>
            </article>

            <div v-if="messages.length === 0" class="empty-state">
              <div class="empty-symbol">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                  <path d="M3 5.5A1.5 1.5 0 0 1 4.5 4h11A1.5 1.5 0 0 1 17 5.5v7a1.5 1.5 0 0 1-1.5 1.5H8l-4 3v-3H4.5A1.5 1.5 0 0 1 3 12.5z" stroke-linejoin="round" />
                  <path d="M6.5 7.5h7M6.5 10.5h4.5" stroke-linecap="round" />
                </svg>
              </div>
              <h2>从一个客户问题开始</h2>
              <p>下面的快捷问题只是起点，你也可以直接输入自己的测试用例。</p>
              <div class="starter-prompts">
                <button @click="usePrompt('我想申请退款，订单号是 #12345')">退款申请</button>
                <button @click="usePrompt('登录时提示错误，应该怎么排查？')">技术排查</button>
                <button @click="usePrompt('发票多久可以开具？')">发票咨询</button>
              </div>
            </div>
          </div>

          <form class="composer" @submit.prevent="sendMessage">
            <textarea
              v-model="draft"
              rows="3"
              placeholder="输入消息..."
              @keydown.meta.enter.prevent="sendMessage"
              @keydown.ctrl.enter.prevent="sendMessage"
            ></textarea>
            <div class="composer-bottom">
              <span class="composer-hint">
                <kbd>Ctrl</kbd>+<kbd>Enter</kbd> 发送
              </span>
              <div class="composer-actions">
                <button v-if="busy" class="quiet-button" @click="cancelChat">取消</button>
                <button type="submit" :disabled="busy || !draft.trim()">{{ busy ? '处理中' : '发送' }}</button>
              </div>
            </div>
          </form>
        </section>

        <aside class="chat-sidebar" ref="sidebarRef">
          <div class="chat-sidebar-scroll">
            <section class="side-card session-card">
              <div class="card-heading">
                <h2>会话信息</h2>
                <span class="status-copy muted">{{ settings.conversationId ? '已启用' : '新会话' }}</span>
              </div>
              <div class="session-grid">
                <div>
                  <span>会话 ID</span>
                  <strong>{{ settings.conversationId || '自动生成' }}</strong>
                </div>
                <div>
                  <span>用户 ID</span>
                  <strong>{{ settings.userId || 'anonymous' }}</strong>
                </div>
              </div>
            </section>

            <section class="side-card connection-card">
              <div class="card-heading">
                <h2>连接配置</h2>
                <span class="status-copy" :class="healthOk ? 'success' : 'muted'">{{ healthLabel }}</span>
              </div>

              <label>
                <span>用户 ID</span>
                <input v-model="settings.userId" @change="persist" placeholder="u1001" />
              </label>
              <label>
                <span>会话 ID</span>
                <input v-model="settings.conversationId" @change="persist" placeholder="自动生成" />
              </label>
              <label>
                <span>访问令牌</span>
                <input v-model="settings.apiToken" @change="persist" type="password" placeholder="留空表示后端未启用鉴权" />
              </label>
              <div class="side-actions">
                <button class="quiet-button" @click="refreshConsole">刷新</button>
              </div>
            </section>

            <section class="side-card trace-card">
              <div class="card-heading">
                <h2>最近一次请求</h2>
                <button v-if="lastRequestId" class="link-button" @click="openTrace(lastRequestId)">查看链路</button>
              </div>

              <div v-if="lastResponse" class="trace-body">
                <div class="latency">
                  <span>响应耗时</span>
                  <strong>{{ lastResponse.latencyMs || '-' }}<small> ms</small></strong>
                </div>
                <dl class="detail-list">
                  <div><dt>主 Agent</dt><dd>{{ lastResponse.primaryAgent || lastResponse.agentType || '-' }}</dd></div>
                  <div><dt>意图</dt><dd>{{ lastResponse.intent || '-' }}</dd></div>
                  <div><dt>置信度</dt><dd>{{ formatPercent(lastResponse.routingConfidence) }}</dd></div>
                  <div><dt>知识库</dt><dd :class="lastResponse.knowledgeUsed ? 'success' : 'muted'">{{ lastResponse.knowledgeUsed ? '已使用' : '未使用' }}</dd></div>
                  <div><dt>降级</dt><dd :class="lastResponse.degraded ? 'warn' : 'muted'">{{ lastResponse.degraded ? `是 · ${lastResponse.degradations.length} 项` : '否' }}</dd></div>
                  <div><dt>转人工</dt><dd :class="lastResponse.escalated ? 'danger' : 'muted'">{{ lastResponse.escalated ? '是' : '否' }}</dd></div>
                </dl>
                <p v-if="lastResponse.routingReason" class="routing-reason">{{ lastResponse.routingReason }}</p>
                <ul v-if="lastResponse.degradations?.length" class="degrade-list">
                  <li v-for="(event, index) in lastResponse.degradations" :key="`${event.source}-${event.code}-${index}`">
                    <code>{{ event.source }}/{{ event.code }}</code>
                    <span>{{ event.message }}</span>
                  </li>
                </ul>
                <div v-if="lastTrace?.trace" class="trace-call-list">
                  <div class="trace-call-title">工具调用</div>
                  <div v-for="(call, index) in lastTrace.trace.toolCalls" :key="`${call.tool_use_id || index}`" class="trace-call-item">
                    <div class="trace-call-meta">
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.latency_ms || 0 }} ms</span>
                    </div>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </div>
                  <div v-if="!lastTrace.trace.toolCalls?.length" class="trace-empty-block">
                    <p>这次 trace 没有记录到工具输入。</p>
                    <p v-if="lastTrace.trace.toolsUsed?.length" class="trace-note">已调用：{{ lastTrace.trace.toolsUsed.join(' · ') }}</p>
                  </div>
                </div>
              </div>
              <p v-else class="side-empty">发送消息后，这里会显示 Agent 路由、意图和耗时。</p>
            </section>

            <section class="side-card monitor-card">
              <div class="card-heading">
                <h2>运行状态</h2>
              </div>
              <div class="mini-stats">
                <div><strong>{{ totalRequests }}</strong><span>请求</span></div>
                <div><strong>{{ agentCount }}</strong><span>Agent</span></div>
                <div><strong>{{ activeAlerts.length }}</strong><span>告警</span></div>
              </div>
              <div v-if="activeAlerts.length" class="alert-note" role="status">
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                  <path d="M8 2.5 14.5 13.5H1.5z" stroke-linejoin="round" />
                  <path d="M8 6.5v3M8 11.5v.5" stroke-linecap="round" />
                </svg>
                <span><span class="alert-severity">{{ activeAlerts[0].severity }}</span>{{ activeAlerts[0].message }}</span>
              </div>
              <p v-else class="healthy-note" role="status">
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                  <circle cx="8" cy="8" r="6" />
                  <path d="m5.5 8 1.8 1.8L10.8 6.3" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                当前没有活跃告警。
              </p>
            </section>
          </div>
        </aside>
      </div>
    </section>

    <section v-else-if="activeView === 'knowledge'" class="page page-knowledge">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">POST /search</span>
          <h1>知识库</h1>
          <p>搜索、补充和维护客服 Agent 使用的知识片段。</p>
        </div>
        <div class="count-display"><strong>{{ knowledgeCount }}</strong><span>chunks</span></div>
      </div>

      <div class="knowledge-layout">
        <section class="workspace-card search-workspace">
          <div class="card-heading">
            <h2>检索知识</h2>
            <code>top_k 5</code>
          </div>
          <div class="search-line">
            <input v-model="searchQuery" placeholder="例如：退款多久到账" @keydown.enter="searchKnowledge" />
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">搜索</button>
          </div>
          <div v-if="searchedOnce" class="search-diag">
            <div class="search-diag-top">
              <span class="stage-chips">
                <span v-for="chip in stageChips" :key="chip.text" :class="chip.tone">{{ chip.text }}</span>
              </span>
              <span v-if="searchError" class="degraded-tag danger-tag">检索失败</span>
              <span v-else-if="searchDegraded" class="degraded-tag">DEGRADED</span>
            </div>
            <p v-if="searchError" class="search-error">{{ searchError }}</p>
            <div v-if="funnelRows.length" class="funnel">
              <div v-for="row in funnelRows" :key="row.label" class="funnel-row">
                <span>{{ row.label }}</span>
                <i><b :style="{ width: `${row.width}%` }"></b></i>
                <strong>{{ row.value }}</strong>
              </div>
            </div>
          </div>
          <div v-if="searchResults.length" class="result-list">
            <article v-for="(item, index) in searchResults" :key="item.id || item.title || index" class="result-item">
              <span class="result-number">{{ String(index + 1).padStart(2, '0') }}</span>
              <div>
                <div class="result-title"><strong>{{ item.title || '未命名文档' }}</strong><small>score {{ item.score ?? '-' }}</small></div>
                <p>{{ item.content }}</p>
              </div>
            </article>
          </div>
          <div v-else class="workspace-empty">{{ searchedOnce ? '这次检索没有命中任何片段。' : '输入客户问题开始搜索。' }}</div>
        </section>

        <section class="workspace-card import-workspace">
          <div class="card-heading">
            <h2>添加知识</h2>
            <code>Milvus 混合索引</code>
          </div>
          <label><span>标题</span><input v-model="docTitle" placeholder="退款补充政策" /></label>
          <label><span>内容</span><textarea v-model="docContent" rows="7" placeholder="输入客服规范、产品说明或排障流程"></textarea></label>
          <div class="side-actions">
            <button @click="submitKnowledge" :disabled="busy || !docTitle.trim() || !docContent.trim()">添加文档</button>
            <label class="upload-button">上传文件<input type="file" accept=".txt,.md,.json" @change="handleUpload" /></label>
          </div>
        </section>
      </div>

      <section class="workspace-card skills-workspace">
        <div class="card-heading">
          <h2>已加载能力 · {{ skillsData.skills.length }}</h2>
          <button class="link-button" @click="reloadSkillSet">重新加载</button>
        </div>
        <div class="skill-table">
          <div v-for="skill in skillsData.skills" :key="skill.name" class="skill-item">
            <span class="skill-dot"></span><strong>{{ skill.name }}</strong><span>{{ skill.description || '业务规范能力' }}</span><small>{{ skill.content_chars || 0 }} chars</small>
          </div>
          <div v-if="!skillsData.skills.length" class="workspace-empty">暂无已加载 Skill。</div>
        </div>
      </section>
    </section>

    <section v-else-if="activeView === 'trace'" class="page page-trace">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">GET /trace/{trace_id}</span>
          <h1>请求链路</h1>
          <p>一次请求的 span 树：编排 → Agent → 工具调用 → RAG 检索，各层谁把时间花在哪。</p>
        </div>
        <div class="heading-actions">
          <span class="session-label">trace: {{ selectedTraceId || 'none' }}</span>
          <button class="quiet-button" @click="loadRecentTraces">刷新列表</button>
        </div>
      </div>

      <div class="trace-layout">
        <section class="workspace-card trace-list-workspace">
          <div class="card-heading">
            <h2>最近请求</h2>
            <code>limit 20</code>
          </div>
          <div v-if="recentTraces.length" class="trace-list">
            <button
              v-for="item in recentTraces"
              :key="item.trace_id"
              class="trace-list-item"
              :class="{ active: item.trace_id === selectedTraceId }"
              @click="openTrace(item.trace_id)"
            >
              <div class="trace-list-top">
                <strong>#{{ item.trace_id }}</strong>
                <span>{{ item.latency_ms }} ms</span>
              </div>
              <div class="trace-list-meta">
                <span>{{ item.span_count }} spans</span>
                <span>{{ item.meta?.primary_agent || item.meta?.agent_type || '-' }}</span>
                <span :class="item.status === 'error' ? 'danger' : 'muted'">{{ item.status }}</span>
              </div>
              <small>{{ item.start_time }}</small>
            </button>
          </div>
          <div v-else class="workspace-empty">还没有 trace。发一条对话或在知识库搜一次就有了。</div>
        </section>

        <section class="workspace-card trace-tree-workspace">
          <div class="card-heading">
            <h2>{{ selectedTraceId ? `#${selectedTraceId}` : '链路详情' }}</h2>
            <div v-if="selectedTrace?.tree" class="trace-tree-summary">
              <span>{{ selectedTrace.tree.span_count }} spans</span>
              <strong>{{ selectedTrace.tree.latency_ms }} ms</strong>
            </div>
          </div>
          <div v-if="traceMeta" class="trace-meta">
            <dl class="detail-list">
              <div>
                <dt>主 Agent</dt>
                <dd>{{ traceMeta.primaryAgent || traceMeta.agentType || '-' }}</dd>
              </div>
              <div>
                <dt>工具</dt>
                <dd>{{ traceMeta.toolsUsed.join(' · ') || '-' }}</dd>
              </div>
              <div v-if="traceMeta.degradations.length">
                <dt>降级</dt>
                <dd class="warn">{{ traceMeta.degradations.length }} 项</dd>
              </div>
            </dl>
            <p v-if="traceMeta.routingReason" class="routing-reason">{{ traceMeta.routingReason }}</p>
            <ul v-if="traceMeta.degradations.length" class="degrade-list">
              <li v-for="(event, index) in traceMeta.degradations" :key="`${event.source}-${event.code}-${index}`">
                <code>{{ event.source }}/{{ event.code }}</code>
                <span>{{ event.message }}</span>
              </li>
            </ul>
          </div>
          <div class="kind-legend" aria-hidden="true">
            <span class="l-orchestrator"><i></i>编排</span>
            <span class="l-agent"><i></i>Agent</span>
            <span class="l-llm"><i></i>LLM</span>
            <span class="l-tool"><i></i>工具</span>
            <span class="l-rag"><i></i>RAG</span>
            <span class="l-memory"><i></i>记忆</span>
          </div>
          <div class="trace-tree-scroll">
            <SpanNode
              v-if="selectedTrace?.found && selectedTrace.tree?.root"
              :span="selectedTrace.tree.root"
              :depth="0"
              :root-latency="selectedTrace.tree.latency_ms"
            />
            <div v-else-if="selectedTraceId && selectedTrace && !selectedTrace.found" class="workspace-empty">
              这条 trace 已超出环形缓冲（后端 OPTISERVE_TRACE_TREE_MAX，默认 200 条）或不存在。
            </div>
            <div v-else class="workspace-empty">从左侧选一条请求。</div>
          </div>
        </section>
      </div>
    </section>

    <section v-else class="page page-evaluation">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">POST /eval/run</span>
          <h1>评测 Agent</h1>
          <p>运行 FastAPI 内置评测，查看意图识别、对话质量和回归结果。</p>
        </div>
        <button @click="runEvaluation" :disabled="busy">{{ busy ? '运行中...' : '运行评测' }}</button>
      </div>

      <div v-if="evalData" class="evaluation-content">
        <div class="evaluation-summary">
          <div class="score-hero"><span>通过率</span><strong>{{ formatPercent(evalData.pass_rate) }}</strong><small>{{ evalData.passed }} / {{ evalData.total }} 条用例通过</small></div>
          <div><span>通过</span><strong>{{ evalData.passed }}</strong></div>
          <div><span>总数</span><strong>{{ evalData.total }}</strong></div>
          <div><span>回归</span><strong :class="evalData.regressions?.length ? 'danger' : 'success'">{{ evalData.regressions?.length || 0 }}</strong></div>
        </div>
        <div class="evaluation-layout">
          <section class="workspace-card">
            <div class="card-heading"><h2>平均评分</h2><code>0 – 1</code></div>
            <div class="score-list">
              <div v-for="(value, key) in evalData.avg_scores" :key="key"><span>{{ key }}</span><i><b :style="{ width: `${Math.min(Number(value) * 100, 100)}%` }"></b></i><strong>{{ Number(value).toFixed(2) }}</strong></div>
            </div>
          </section>
          <section class="workspace-card">
            <div class="card-heading"><h2>优化建议</h2><code>{{ evalData.recommendations?.length || 0 }} 条</code></div>
            <div v-if="evalData.recommendations?.length" class="recommendations"><p v-for="(item, index) in evalData.recommendations" :key="index">{{ item }}</p></div>
            <div v-else class="workspace-empty">本次评测没有返回额外建议。</div>
          </section>
        </div>
      </div>
      <div v-else class="evaluation-empty">
        <div class="empty-symbol">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
            <circle cx="10" cy="10" r="7" />
            <path d="M8.2 7.4 13 10l-4.8 2.6z" stroke-linejoin="round" />
          </svg>
        </div>
        <h2>还没有评测结果</h2>
        <p>点击右上角运行一次评测。</p>
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import SpanNode from './components/SpanNode.vue'
import {
  API_BASE,
  addKnowledge,
  createInitialSettings,
  reloadSkills,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestRecentTraces,
  requestSearch,
  requestToolTrace,
  requestTraceTree,
  requestSkills,
  runEvaluation as requestEvaluation,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const settings = reactive(createInitialSettings())
const activeView = ref('chat')
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const knowledgeCount = ref('-')
const searchQuery = ref('退款多久能到账')
const searchResults = ref([])
const searchStages = ref({})
const searchDegraded = ref(false)
const searchError = ref('')
const searchedOnce = ref(false)
const docTitle = ref('退款补充政策')
const docContent = ref('大促期间退款审核时间可能延长到 3-5 个工作日。')
const messageList = ref(null)
const sidebarRef = ref(null)
const monitorData = ref({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
const skillsData = ref({ count: 0, skills: [], errors: [] })
const lastResponse = ref(null)
const lastTrace = ref(null)
const lastRequestId = ref('')
const recentTraces = ref([])
const selectedTraceId = ref('')
const selectedTrace = ref(null)
const evalData = ref(null)
const toast = ref('')
let toastTimer
let messageSequence = 0
let sidebarObserver

const docsUrl = computed(() => `${API_BASE}/docs`)
const userInitial = computed(() => (settings.userId || 'U').slice(0, 1).toUpperCase())
const activeAlerts = computed(() => monitorData.value.active_alerts || [])
const agentCount = computed(() => Object.keys(monitorData.value.agent_stats || {}).length)
const totalRequests = computed(() => Object.values(monitorData.value.agent_stats || {}).reduce((sum, item) => sum + Number(item.total || 0), 0))

/* /search 的 stages 是「各级还剩几条」，只有粗排之后的三级是同一量纲的文档条数，
   所以它们画成漏斗；改写条数与召回路数是另一种单位，只做旁注。 */
const funnelRows = computed(() => {
  const stages = searchStages.value || {}
  const rows = [
    { label: '粗排候选', key: 'coarse' },
    { label: '精排', key: 'reranked' },
    { label: '截断返回', key: 'returned' }
  ]
    .filter(row => typeof stages[row.key] === 'number')
    .map(row => ({ label: row.label, value: stages[row.key] }))
  const max = Math.max(1, ...rows.map(row => row.value))
  return rows.map(row => ({ ...row, width: Math.max(2, (row.value / max) * 100) }))
})

const stageChips = computed(() => {
  const stages = searchStages.value || {}
  const chips = []
  if (typeof stages.rewrite === 'number') chips.push({ text: `改写 ${stages.rewrite} 条子查询` })
  if (typeof stages.recall_paths === 'number') chips.push({ text: `${stages.recall_paths} 路混合召回` })
  if (stages.recall_failed) chips.push({ text: `${stages.recall_failed} 路召回失败`, tone: 'warn' })
  return chips
})

// span 树的 meta 由 API 层写入，只有 /chat 那一路有；/search 的 trace 这里就是空的
const traceMeta = computed(() => {
  const meta = selectedTrace.value?.tree?.meta
  if (!meta || !Object.keys(meta).length) return null
  return {
    agentType: meta.agent_type || '',
    primaryAgent: meta.primary_agent || '',
    routingReason: meta.routing_reason || '',
    toolsUsed: meta.tools_used || [],
    escalated: Boolean(meta.escalated),
    degradations: meta.degradations || []
  }
})

onMounted(() => {
  refreshConsole()
  updateSidebarHeight()
  if (typeof ResizeObserver !== 'undefined') {
    sidebarObserver = new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.value) sidebarObserver.observe(sidebarRef.value)
  }
  window.addEventListener('resize', updateSidebarHeight)
})

onBeforeUnmount(() => {
  sidebarObserver?.disconnect?.()
  window.removeEventListener('resize', updateSidebarHeight)
  consoleController?.abort()
  chatController?.abort()
})

function persist() { saveSettings(settings) }

function updateSidebarHeight() {
  const sidebar = sidebarRef.value
  if (!sidebar) return
  const rect = sidebar.getBoundingClientRect()
  const height = Math.max(320, Math.floor(rect.height))
  sidebar.style.setProperty('--sidebar-height', `${height}px`)
}

// 刷新即放弃上一轮：连点刷新时旧请求回来得晚，会把上一轮数据盖回界面上
let consoleController = null
let chatController = null

async function refreshConsole() {
  consoleController?.abort()
  const controller = new AbortController()
  consoleController = controller
  const { signal } = controller
  await Promise.allSettled([
    checkHealth(signal),
    loadStats(signal),
    loadMonitor(signal),
    loadSkills(signal),
    loadRecentTraces(signal)
  ])
}

async function checkHealth(signal) {
  try {
    const data = await requestHealth(signal)
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
  } catch (error) {
    if (error.cancelled) return
    healthOk.value = false
    healthLabel.value = '不可用'
    showToast(`后端不可用：${error.message}`)
  }
}

async function loadStats(signal) {
  try {
    const data = await requestKnowledgeStats(signal)
    knowledgeCount.value = data.total_chunks ?? '-'
  } catch {
    knowledgeCount.value = '-'
  }
}

async function loadMonitor(signal) {
  try {
    monitorData.value = await requestMonitor(signal)
  } catch (error) {
    if (!error.cancelled) monitorData.value = { agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] }
  }
}

async function loadSkills(signal) {
  try {
    skillsData.value = await requestSkills(signal)
  } catch (error) {
    if (!error.cancelled) skillsData.value = { count: 0, skills: [], errors: [] }
  }
}

async function reloadSkillSet() {
  busy.value = true
  try {
    skillsData.value = await reloadSkills()
    showToast('Skills 已重新加载')
  } catch (error) {
    showToast(`Skills 加载失败：${error.message}`)
  } finally { busy.value = false }
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || busy.value) return
  messages.value.push({ id: createMessageId(), role: 'user', content })
  draft.value = ''
  busy.value = true
  const controller = new AbortController()
  chatController = controller
  try {
    const response = await requestChat(settings, content, controller.signal)
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    lastResponse.value = response
    lastRequestId.value = response.requestId
    lastTrace.value = await loadToolTrace(response.requestId)
    const meta = [response.intent, response.primaryAgent || response.agentType, response.knowledgeUsed ? 'RAG' : '', response.escalated ? '转人工' : ''].filter(Boolean).join(' · ')
    messages.value.push({
      id: createMessageId(),
      role: 'assistant',
      content: response.response,
      meta,
      degraded: response.degraded,
      degradations: response.degradations
    })
    await Promise.allSettled([loadMonitor(), loadRecentTraces()])
  } catch (error) {
    messages.value.push({
      id: createMessageId(),
      role: 'assistant',
      content: error.cancelled ? '已取消这次请求。' : error.message,
      meta: error.cancelled ? '已取消' : error.timedOut ? '请求超时' : '请求失败',
      degraded: false,
      degradations: []
    })
    if (error.timedOut) showToast(error.message)
  } finally {
    chatController = null
    busy.value = false
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

function cancelChat() {
  chatController?.abort()
}

function usePrompt(prompt) { draft.value = prompt }

function clearConversation() {
  messages.value = []
  lastResponse.value = null
  lastTrace.value = null
  lastRequestId.value = ''
  settings.conversationId = ''
  persist()
}

function openTraceView() {
  activeView.value = 'trace'
  if (!selectedTraceId.value) {
    selectedTraceId.value = lastRequestId.value
    if (selectedTraceId.value) openTrace(selectedTraceId.value)
  }
}

async function openTrace(traceId) {
  activeView.value = 'trace'
  selectedTraceId.value = traceId
  selectedTrace.value = null
  try {
    selectedTrace.value = await requestTraceTree(traceId)
  } catch (error) {
    showToast(`链路读取失败：${error.message}`)
  }
}

async function loadRecentTraces(signal) {
  try {
    const data = await requestRecentTraces(20, signal)
    recentTraces.value = data.items || []
    if (!selectedTraceId.value && recentTraces.value.length) {
      selectedTraceId.value = recentTraces.value[0].trace_id
      selectedTrace.value = await requestTraceTree(selectedTraceId.value)
    }
  } catch (error) {
    if (!error.cancelled) recentTraces.value = []
  }
}

async function searchKnowledge() {
  busy.value = true
  try {
    const data = await requestSearch(searchQuery.value, 5)
    searchResults.value = data.results
    searchStages.value = data.stages
    searchDegraded.value = data.degraded
    searchError.value = data.error
    searchedOnce.value = true
    if (data.error) {
      showToast(`检索失败：${data.error}`)
    } else {
      showToast(`检索完成，返回 ${data.results.length} 条结果${data.degraded ? '（已降级）' : ''}`)
    }
    loadRecentTraces()
  } catch (error) {
    searchResults.value = []
    searchStages.value = {}
    searchDegraded.value = false
    searchError.value = error.message
    searchedOnce.value = true
    showToast(`检索失败：${error.message}`)
  } finally { busy.value = false }
}

async function submitKnowledge() {
  busy.value = true
  try {
    await addKnowledge([{ title: docTitle.value.trim(), content: docContent.value.trim() }])
    await loadStats()
    showToast('文档已添加')
  } catch (error) {
    showToast(`文档导入失败：${error.message}`)
  } finally { busy.value = false }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busy.value = true
  try {
    await uploadKnowledge(file)
    await loadStats()
    showToast(`${file.name} 导入成功`)
  } catch (error) {
    showToast(`文件导入失败：${error.message}`)
  } finally { busy.value = false }
}

async function runEvaluation() {
  busy.value = true
  try {
    evalData.value = await requestEvaluation()
    showToast('评测完成')
  } catch (error) {
    showToast(`评测运行失败：${error.message}`)
  } finally { busy.value = false }
}

async function loadToolTrace(requestId) {
  try {
    return await requestToolTrace(requestId)
  } catch {
    return null
  }
}

function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
}

function formatJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return String(value ?? '')
  }
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}
</script>
