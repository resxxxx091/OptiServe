export const API_BASE = String(import.meta.env.VITE_PYTHON_API_URL || '/api/python').replace(/\/+$/, '')

/* 检索只有一处 top_k：界面文案和请求参数都读它。 */
export const SEARCH_TOP_K = 5

/* Langfuse 项目首页，形如 https://<host>/project/<projectId>；留空则界面不出跳转链接。
   只到项目页不到单条 trace：trace_id 由 request_id seed 派生（后端 create_trace_id），
   前端拿不到也没法反推，精确深链得后端回传那个 32 位 ID。 */
export const LANGFUSE_PROJECT_URL = String(import.meta.env.VITE_LANGFUSE_PROJECT_URL || '').replace(/\/+$/, '')

/* 用户 ID 空值口径只有一个：界面链接和 /chat 请求体都走它，否则 Langfuse 筛不到。 */
export function chatUserId(settings) {
  return settings.userId || 'anonymous'
}

const SETTINGS_KEY = 'optiserve.frontend.settings'

/* 这一层只是最外层保险：必须大于后端的对应预算，否则后端还没来得及降级、界面先报错。
   后端预算是 RETRIEVAL_TOTAL_TIMEOUT_S=45s、OPTISERVE_AGENT_LOOP_TIMEOUT_S=90s。 */
export const TIMEOUT = {
  read: 10000,
  search: 60000,
  write: 120000,
  chat: 150000
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    apiToken: saved.apiToken || ''
  }
}

export function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
}

export function requestHealth(signal) {
  return requestJson('/health', { signal }).then(normalizeHealthResponse)
}

export function requestMonitor(signal) {
  return requestJson('/monitor', { signal })
}

export function requestSkills(signal) {
  return requestJson('/skills', { signal })
}

export function reloadSkills(signal) {
  return requestJson('/skills/reload', { method: 'POST', signal, timeoutMs: TIMEOUT.write })
}

export function requestKnowledgeStats(signal) {
  return requestJson('/knowledge/stats', { signal })
}

export function requestSearch(query, signal) {
  const params = new URLSearchParams({ query, top_k: String(SEARCH_TOP_K) })
  return requestJson(`/search?${params}`, { method: 'POST', signal, timeoutMs: TIMEOUT.search }).then(
    normalizeSearchResponse
  )
}

export function requestChat(settings, message, signal) {
  return requestJson('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: chatUserId(settings),
      conv_id: settings.conversationId || undefined
    }),
    signal,
    timeoutMs: TIMEOUT.chat
  }).then(normalizeChatResponse)
}

export function addKnowledge(documents) {
  return requestJson('/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents }),
    timeoutMs: TIMEOUT.write
  })
}

export function uploadKnowledge(file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson('/knowledge/upload', { method: 'POST', body: form, timeoutMs: TIMEOUT.write })
}

/* 只做 snake_case → camelCase 改名：/chat 挂了 response_model=ChatResponse，
   200 响应里每个字段都必在且类型已定，这里再兜一层只会把契约问题咽掉。 */
function normalizeChatResponse(raw) {
  return {
    requestId: raw.request_id,
    conversationId: raw.conv_id,
    response: raw.response,
    intent: raw.intent,
    primaryAgent: raw.primary_agent,
    toolsUsed: raw.tools_used,
    routingReason: raw.routing_reason,
    routingConfidence: raw.routing_confidence,
    escalated: raw.escalated,
    latencyMs: raw.latency_ms,
    knowledgeUsed: raw.knowledge_used,
    degradations: raw.degradations,
    degraded: raw.degraded
  }
}

function normalizeSearchResponse(raw) {
  return {
    results: raw.results || [],
    degraded: Boolean(raw.degraded),
    error: raw.error || '',
    stages: raw.stages || {}
  }
}

/* /health 的 dependencies 是启动闸门逐项探测的结果：state 只会是 ok（不通的那一项会让服务
   直接拒绝启动），detail 是真实接上的端点与条数摘要。unhealthy 预留给将来的运行期探活。 */
function normalizeHealthResponse(raw) {
  const dependencies = Object.entries(raw?.dependencies || {}).map(([source, info]) => ({
    source,
    state: info?.state || 'unknown',
    detail: info?.detail || ''
  }))
  return {
    status: raw?.status || 'ok',
    dependencies,
    unhealthy: dependencies.filter(item => item.state !== 'ok')
  }
}

async function requestJson(path, options = {}) {
  const { timeoutMs = TIMEOUT.read, signal, ...fetchOptions } = options
  // 合并而非覆盖：调用方已带 Content-Type，上传接口刻意不带以便浏览器生成 FormData 边界
  const headers = new Headers(fetchOptions.headers)
  const token = readSettings().apiToken
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      headers,
      signal: composeSignal(signal, timeoutMs)
    })
  } catch (error) {
    // 后端挂住时 fetch 只会一直等，这里让它到点变成一句能读的提示
    if (error?.name === 'AbortError' && signal?.aborted) {
      throw Object.assign(new Error('请求已取消'), { cancelled: true })
    }
    if (error?.name === 'AbortError' || error?.name === 'TimeoutError') {
      throw Object.assign(new Error(`后端未在 ${timeoutMs / 1000}s 内返回`), { timedOut: true })
    }
    throw new Error(`网络错误：${error?.message || error}`)
  }
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw Object.assign(new Error(`${response.status} ${response.statusText}: ${detail}`), {
      status: response.status
    })
  }
  return data
}

function composeSignal(signal, timeoutMs) {
  const timeout = AbortSignal.timeout(timeoutMs)
  return signal ? AbortSignal.any([signal, timeout]) : timeout
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
  } catch {
    return {}
  }
}
