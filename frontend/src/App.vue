<template>
  <main :class="['app-shell', `app-shell-${activeView}`]">
    <header class="topbar">
      <a class="brand" href="#" aria-label="OptiServe 首页" @click.prevent="activeView = 'chat'">
        <span class="brand-mark">OS</span>
        <span class="brand-name">OptiServe<small>调试页面</small></span>
      </a>

      <nav class="view-nav" aria-label="工作区">
        <button :class="{ active: activeView === 'chat' }" @click="activeView = 'chat'">对话</button>
        <button :class="{ active: activeView === 'knowledge' }" @click="activeView = 'knowledge'">知识库</button>
      </nav>

      <div class="topbar-tools">
        <span class="environment-pill" :title="dependencyHint">
          <i :class="healthDot"></i>
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
        <button
          class="icon-button"
          :class="{ active: settingsOpen }"
          title="设置"
          aria-label="设置"
          :aria-expanded="settingsOpen"
          aria-haspopup="dialog"
          @click="toggleSettings"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
            <circle cx="12" cy="12" r="3.2" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke-linejoin="round" />
          </svg>
        </button>
      </div>
    </header>

    <Transition name="toast">
      <div v-if="toast" class="toast" role="status">{{ toast }}</div>
    </Transition>

    <Transition name="modal">
      <div v-if="settingsOpen" class="modal-backdrop" @click.self="settingsOpen = false">
      <section class="modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header class="modal-head">
          <h2 id="settings-title">设置</h2>
          <button class="icon-button" aria-label="关闭设置" @click="settingsOpen = false">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
              <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
            </svg>
          </button>
        </header>

        <div class="modal-body">
          <section class="modal-section">
            <div class="card-heading">
              <h3>身份与会话</h3>
              <span class="status-copy muted">写入 localStorage</span>
            </div>
            <label>
              <span>用户 ID</span>
              <input ref="userIdInput" v-model="settings.userId" @change="persist" placeholder="u1001" />
            </label>
            <label>
              <span>会话 ID</span>
              <input v-model="settings.conversationId" @change="persist" placeholder="留空则每次新生成" />
            </label>
            <label>
              <span>访问令牌</span>
              <input v-model="settings.apiToken" @change="persist" type="password" placeholder="留空则写入类操作会被后端拒绝" />
            </label>
          </section>

          <section class="modal-section">
            <div class="card-heading">
              <h3>后端与请求</h3>
              <code>只读</code>
            </div>
            <dl class="detail-list">
              <div><dt>API 基址</dt><dd>{{ API_BASE }}</dd></div>
              <div><dt>检索 top_k</dt><dd>{{ SEARCH_TOP_K }}</dd></div>
              <div><dt>读超时</dt><dd>{{ TIMEOUT.read / 1000 }} s</dd></div>
              <div><dt>检索超时</dt><dd>{{ TIMEOUT.search / 1000 }} s</dd></div>
              <div><dt>写入超时</dt><dd>{{ TIMEOUT.write / 1000 }} s</dd></div>
              <div><dt>对话超时</dt><dd>{{ TIMEOUT.chat / 1000 }} s</dd></div>
              <div><dt>Langfuse</dt><dd>{{ LANGFUSE_PROJECT_URL || '未配置' }}</dd></div>
            </dl>
            <p class="modal-note">
              只读项分别来自 <code>src/lib/backends.js</code> 与 <code>VITE_PYTHON_API_URL</code> /
              <code>VITE_LANGFUSE_PROJECT_URL</code>，改完要重启 dev server。
            </p>
          </section>
        </div>
      </section>
    </div>
    </Transition>

    <section v-if="activeView === 'chat'" class="page page-chat">
      <div class="page-heading">
        <div>
          <span class="kicker">POST /chat</span>
          <h1>和运营服务 Agent 对话</h1>
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
            <span>{{ messages.length }} 条消息</span>
            <button v-if="busy" class="quiet-button" @click="cancelChat">取消</button>
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
                <small v-if="item.time" class="msg-time">{{ item.time }}</small>
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

            <div v-if="busy" class="message assistant typing-row" aria-live="polite">
              <div class="message-meta"><span>agent</span><small>处理中</small></div>
              <div class="typing-dots"><i></i><i></i><i></i></div>
            </div>

            <div v-if="messages.length === 0 && !busy" class="empty-state">
              <div class="empty-symbol">
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                  <path d="M3 5.5A1.5 1.5 0 0 1 4.5 4h11A1.5 1.5 0 0 1 17 5.5v7a1.5 1.5 0 0 1-1.5 1.5H8l-4 3v-3H4.5A1.5 1.5 0 0 1 3 12.5z" stroke-linejoin="round" />
                  <path d="M6.5 7.5h7M6.5 10.5h4.5" stroke-linecap="round" />
                </svg>
              </div>
              <h2>有什么我能帮你的吗？</h2>
              <p>从下面的示例开始，或直接输入你的问题。</p>
              <div class="quick-prompts">
                <button v-for="prompt in quickPrompts" :key="prompt" class="quick-prompt" @click="applyPrompt(prompt)">
                  {{ prompt }}
                </button>
              </div>
            </div>
          </div>

          <form class="composer" @submit.prevent="sendMessage">
            <textarea
              ref="draftInput"
              v-model="draft"
              rows="3"
              placeholder="输入消息..."
              @keydown.enter.exact="handleComposerEnter"
            ></textarea>
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

            <section class="side-card trace-card">
              <div class="card-heading">
                <h2>最近一次请求</h2>
              </div>

              <div v-if="lastResponse" class="trace-body">
                <div class="latency">
                  <span>响应耗时</span>
                  <strong>{{ lastResponse.latencyMs ?? '-' }}<small> ms</small></strong>
                </div>
                <dl class="detail-list">
                  <div><dt>主 Agent</dt><dd>{{ lastResponse.primaryAgent }}</dd></div>
                  <div><dt>意图</dt><dd>{{ lastResponse.intent }}</dd></div>
                  <div><dt>置信度</dt><dd>{{ formatPercent(lastResponse.routingConfidence) }}</dd></div>
                  <div v-if="lastResponse.toolsUsed.length"><dt>工具</dt><dd>{{ lastResponse.toolsUsed.join(' · ') }}</dd></div>
                  <div><dt>知识库</dt><dd :class="lastResponse.knowledgeUsed ? 'success' : 'muted'">{{ lastResponse.knowledgeUsed ? '已使用' : '未使用' }}</dd></div>
                  <div><dt>降级</dt><dd :class="lastResponse.degraded ? 'warn' : 'muted'">{{ lastResponse.degraded ? `是 · ${lastResponse.degradations.length} 项` : '否' }}</dd></div>
                  <div><dt>转人工</dt><dd :class="lastResponse.escalated ? 'danger' : 'muted'">{{ lastResponse.escalated ? '是' : '否' }}</dd></div>
                  <div class="trace-row">
                    <dt>request_id</dt>
                    <dd>
                      <span>{{ lastResponse.requestId }}</span>
                      <button class="link-button" @click="copyRequestId">{{ copied ? '已复制' : '复制' }}</button>
                      <a v-if="langfuseUserUrl" class="link-button" :href="langfuseUserUrl" target="_blank" rel="noreferrer">Langfuse</a>
                    </dd>
                  </div>
                </dl>
                <p v-if="lastResponse.routingReason" class="routing-reason">{{ lastResponse.routingReason }}</p>
                <ul v-if="lastResponse.degradations?.length" class="degrade-list">
                  <li v-for="(event, index) in lastResponse.degradations" :key="`${event.source}-${event.code}-${index}`">
                    <code>{{ event.source }}/{{ event.code }}</code>
                    <span>{{ event.message }}</span>
                  </li>
                </ul>
              </div>
              <p v-else class="side-empty">发送消息后，这里会显示 Agent 路由、意图和耗时。</p>
            </section>

            <section class="side-card">
              <div class="card-heading">
                <h2>运行状态</h2>
                <div class="heading-actions">
                  <!-- 有读数时报条数；拿不到时退回健康文案，别在探测失败后谎称「未检查」 -->
                  <span class="status-copy" :class="healthOk ? 'success' : 'muted'">
                    {{ healthDeps.length ? `${healthDeps.length} 项依赖` : healthLabel }}
                  </span>
                  <button class="link-button" @click="refreshConsole">刷新</button>
                </div>
              </div>
              <div class="mini-stats">
                <div><strong>{{ totalRequests }}</strong><span>请求</span></div>
                <div><strong>{{ agentCount }}</strong><span>Agent</span></div>
                <div><strong>{{ toolCount }}</strong><span>工具</span></div>
              </div>
              <dl v-if="healthDeps.length" class="dep-list">
                <div v-for="dep in healthDeps" :key="dep.source">
                  <dt :class="dep.state === 'ok' ? 'success' : 'warn'">{{ dep.source }}</dt>
                  <dd :title="dep.detail || dep.state">{{ dep.detail || dep.state }}</dd>
                </div>
              </dl>
              <div v-if="runtimeChips.length" class="runtime-chips">
                <span v-for="chip in runtimeChips" :key="chip.key" :class="chip.tone" :title="chip.title">{{ chip.text }}</span>
              </div>
            </section>
          </div>
        </aside>
      </div>
    </section>

    <section v-else-if="activeView === 'knowledge'" class="page page-knowledge">
      <div class="page-heading">
        <div>
          <span class="kicker">POST /search</span>
          <h1>知识库</h1>
          <p>搜索、补充和维护运营服务 Agent 使用的知识片段。</p>
        </div>
        <div class="count-display"><strong>{{ knowledgeCount }}</strong><span>chunks</span></div>
      </div>

      <div class="knowledge-layout">
        <section class="workspace-card">
          <div class="card-heading">
            <h2>检索知识</h2>
            <code>top_k {{ SEARCH_TOP_K }}</code>
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

        <section class="workspace-card">
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
            <span class="skill-dot"></span><strong>{{ skill.name }}</strong><span>{{ skill.description }}</span><small>{{ skill.content_chars }} chars</small>
          </div>
          <div v-if="!skillsData.skills.length" class="workspace-empty">暂无已加载 Skill。</div>
        </div>
        <!-- 解析失败后端只进日志和这个字段，界面上不列就永远看不到 -->
        <ul v-if="skillsData.errors.length" class="degrade-list">
          <li v-for="(error, index) in skillsData.errors" :key="`skill-error-${index}`">
            <code>skill</code>
            <span>{{ error }}</span>
          </li>
        </ul>
      </section>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import {
  API_BASE,
  LANGFUSE_PROJECT_URL,
  SEARCH_TOP_K,
  TIMEOUT,
  addKnowledge,
  chatUserId,
  createInitialSettings,
  reloadSkills,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestSkills,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const settings = reactive(createInitialSettings())
const activeView = ref('chat')
const settingsOpen = ref(false)
const userIdInput = ref(null)
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const healthDeps = ref([])
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
const draftInput = ref(null)
/* 初值就按 /monitor 的契约形状给全：三个键后端恒返回，读到空壳而不是 undefined，
   下面就不必为"还没刷过"再兜一层。 */
const monitorData = ref({ agent_stats: {}, tool_stats: {}, routing: { agents: {}, demote_threshold: 0 } })
const skillsData = ref({ skills: [], errors: [] })
const lastResponse = ref(null)
const copied = ref(false)
const toast = ref('')
let toastTimer
let copyTimer
let messageSequence = 0
let sidebarObserver
let chatController = null

const docsUrl = computed(() => `${API_BASE}/docs`)

/* 空态示例：让首屏有可点的入口，点击即填入输入框 */
const quickPrompts = [
  '退款多久能到账？',
  '发票怎么开具？',
  '物流地址可以修改吗？'
]
const agentCount = computed(() => Object.keys(monitorData.value.agent_stats).length)
const toolCount = computed(() => Object.keys(monitorData.value.tool_stats).length)
const totalRequests = computed(() => Object.values(monitorData.value.agent_stats).reduce((sum, item) => sum + item.total, 0))
const langfuseUserUrl = computed(() => {
  if (!LANGFUSE_PROJECT_URL) return ''
  return `${LANGFUSE_PROJECT_URL}/users/${encodeURIComponent(chatUserId(settings))}`
})

// 闸门探测的依赖全通才会亮绿；出现非 ok 状态（将来的运行期探活）转琥珀
const healthDot = computed(() => {
  if (!healthOk.value) return 'offline'
  return healthDeps.value.some(item => item.state !== 'ok') ? 'degraded' : 'online'
})
const dependencyHint = computed(() =>
  healthDeps.value.length
    ? healthDeps.value.map(item => `${item.source}: ${item.state}`).join('\n')
    : '未取到依赖探测结果'
)

/* /monitor 的 tool_stats 与 routing 是"此刻还不对"的状态：熔断器离开 closed、Agent 被
   Monitor 降权。两者都为空时整块不渲染。 */
const runtimeChips = computed(() => {
  const chips = []
  for (const [name, stat] of Object.entries(monitorData.value.tool_stats)) {
    if (stat.circuit_state === 'closed') continue
    chips.push({
      key: `tool-${name}`,
      text: `${name} · ${stat.circuit_state}`,
      tone: stat.circuit_state === 'open' ? 'chip-err' : 'chip-warn',
      title: `熔断 ${stat.circuit_state}：连续失败 ${stat.consecutive_fails} 次，成功率 ${formatPercent(stat.success_rate)}`
    })
  }
  const routing = monitorData.value.routing
  for (const [name, info] of Object.entries(routing.agents)) {
    if (info.penalty <= 0) continue
    chips.push({
      key: `agent-${name}`,
      text: `${name} · 降权 ${info.penalty}`,
      tone: info.demoted ? 'chip-err' : 'chip-warn',
      title: `路由 penalty ${info.penalty}，改判阈值 ${routing.demote_threshold}${info.demoted ? '（已越线：路由会改选意图合法的备选 Agent）' : ''}`
    })
  }
  return chips
})

/* /search 的 stages 是「各级还剩几条」，只有粗排之后的三级是同一量纲的文档条数，
   所以它们画成漏斗；改写条数与召回路数是另一种单位，只做旁注。 */
const funnelRows = computed(() => {
  const stages = searchStages.value
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
  const stages = searchStages.value
  const chips = []
  if (typeof stages.rewrite === 'number') chips.push({ text: `改写 ${stages.rewrite} 条子查询` })
  if (typeof stages.recall_paths === 'number') chips.push({ text: `${stages.recall_paths} 路混合召回` })
  if (stages.recall_failed) chips.push({ text: `${stages.recall_failed} 路召回失败`, tone: 'warn' })
  return chips
})

onMounted(() => {
  refreshConsole()
  updateSidebarHeight()
  if (typeof ResizeObserver !== 'undefined') {
    sidebarObserver = new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.value) sidebarObserver.observe(sidebarRef.value)
  }
  window.addEventListener('resize', updateSidebarHeight)
  window.addEventListener('keydown', onGlobalKeydown)
})

onBeforeUnmount(() => {
  sidebarObserver?.disconnect?.()
  window.removeEventListener('resize', updateSidebarHeight)
  window.removeEventListener('keydown', onGlobalKeydown)
  releaseAll()
})

function persist() { saveSettings(settings) }

function toggleSettings() {
  settingsOpen.value = !settingsOpen.value
  if (settingsOpen.value) nextTick(() => userIdInput.value?.focus())
}

function onGlobalKeydown(event) {
  if (event.key === 'Escape' && settingsOpen.value) settingsOpen.value = false
}

/* 侧栏高度钉成所在行高：行内放不下时走内部滚动。
   不能量侧栏自身——它的兜底高度就是视口推导值，量到什么写回什么，底部卡片会被裁出视口。 */
function updateSidebarHeight() {
  const sidebar = sidebarRef.value
  if (!sidebar) return
  const available = sidebar.parentElement.clientHeight
  sidebar.style.setProperty('--sidebar-height', `${Math.max(320, Math.floor(available))}px`)
}

/* 每个读数一把控制器：begin 会作废上一轮，所以 signal.aborted 就等于"我已经不是最新一轮"。
   被作废的那轮既不写值也不清空，否则连点刷新、或发完消息补一次 /monitor 时，
   迟到的失败会把刚拿到的好数据盖成空的。 */
const inflight = {}
function begin(key) {
  inflight[key]?.abort()
  const controller = new AbortController()
  inflight[key] = controller
  return controller.signal
}

async function refreshConsole() {
  await Promise.allSettled([checkHealth(), loadStats(), loadMonitor(), loadSkills()])
}

async function checkHealth() {
  const signal = begin('health')
  try {
    const data = await requestHealth(signal)
    if (signal.aborted) return
    healthDeps.value = data.dependencies
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.unhealthy.length ? `${data.status} · ${data.unhealthy.length} 项降级` : data.status
  } catch (error) {
    if (signal.aborted || error.cancelled) return
    healthOk.value = false
    healthLabel.value = '不可用'
    healthDeps.value = []
    showToast(`后端不可用：${error.message}`)
  }
}

async function loadStats() {
  const signal = begin('stats')
  try {
    const data = await requestKnowledgeStats(signal)
    if (!signal.aborted) knowledgeCount.value = data.total_chunks
  } catch {
    // 读数保持上一次的值
  }
}

async function loadMonitor() {
  const signal = begin('monitor')
  try {
    const data = await requestMonitor(signal)
    if (!signal.aborted) monitorData.value = data
  } catch {
    // 失败不清空：/monitor 挂了只是没有新数据，把上一次的读数抹成 0 更误导
  }
}

async function loadSkills() {
  const signal = begin('skills')
  try {
    const data = await requestSkills(signal)
    if (!signal.aborted) skillsData.value = data
  } catch {
    // 同上
  }
}

async function reloadSkillSet() {
  busy.value = true
  const signal = begin('skills')
  try {
    const data = await reloadSkills(signal)
    if (!signal.aborted) skillsData.value = data
    showToast('Skills 已重新加载')
  } catch (error) {
    if (!signal.aborted) showToast(`Skills 加载失败：${error.message}`)
  } finally { busy.value = false }
}

/* 中文输入法下 Enter 先用于选词上屏：isComposing / keyCode 229 时不拦截，让组合正常完成 */
function handleComposerEnter(event) {
  if (event.isComposing || event.keyCode === 229) return
  event.preventDefault()
  sendMessage()
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || busy.value) return
  messages.value.push({ id: createMessageId(), role: 'user', content, time: formatTime() })
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
    const meta = [response.intent, response.primaryAgent, response.knowledgeUsed ? 'RAG' : '', response.escalated ? '转人工' : ''].filter(Boolean).join(' · ')
    messages.value.push({
      id: createMessageId(),
      role: 'assistant',
      content: response.response,
      meta,
      time: formatTime(),
      degraded: response.degraded,
      degradations: response.degradations
    })
    await loadMonitor()
  } catch (error) {
    messages.value.push({
      id: createMessageId(),
      role: 'assistant',
      content: error.cancelled ? '已取消这次请求。' : error.message,
      meta: error.cancelled ? '已取消' : error.timedOut ? '请求超时' : '请求失败',
      time: formatTime(),
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

function applyPrompt(text) {
  draft.value = text
  draftInput.value?.focus()
}

function clearConversation() {
  messages.value = []
  lastResponse.value = null
  settings.conversationId = ''
  persist()
}

async function searchKnowledge() {
  // 函数内收口：按钮的 disabled 挡不住输入框的回车
  if (busy.value || !searchQuery.value.trim()) return
  busy.value = true
  const signal = begin('search')
  try {
    const data = await requestSearch(searchQuery.value, signal)
    if (signal.aborted) return
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
  } catch (error) {
    if (signal.aborted) return
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
    const data = await addKnowledge([{ title: docTitle.value.trim(), content: docContent.value.trim() }])
    knowledgeCount.value = data.total_chunks
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
    const data = await uploadKnowledge(file)
    knowledgeCount.value = data.total_chunks
    showToast(`${file.name} 导入成功`)
  } catch (error) {
    showToast(`文件导入失败：${error.message}`)
  } finally { busy.value = false }
}

async function copyRequestId() {
  try {
    await navigator.clipboard.writeText(lastResponse.value.requestId)
    copied.value = true
    clearTimeout(copyTimer)
    copyTimer = setTimeout(() => { copied.value = false }, 1600)
  } catch {
    showToast('复制失败，浏览器未授权剪贴板')
  }
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

/* 调试工作台的时间戳只到秒：对日志足够，也不挤占 meta 行 */
function formatTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false })
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}

function releaseAll() {
  clearTimeout(toastTimer)
  clearTimeout(copyTimer)
  chatController?.abort()
  Object.values(inflight).forEach(controller => controller?.abort())
}
</script>
