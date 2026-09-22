export const API_BASE = String(import.meta.env.VITE_PYTHON_API_URL || '/api/python').replace(/\/+$/, '')

const SETTINGS_KEY = 'optiserve.frontend.settings'

/* 这一层只是最外层保险：必须大于后端的对应预算，否则后端还没来得及降级、界面先报错。
   后端预算是 RETRIEVAL_TOTAL_TIMEOUT_S=45s、OPTISERVE_AGENT_LOOP_TIMEOUT_S=90s。 */
const TIMEOUT = {
  read: 10000,
  search: 60000,
  write: 120000,
  chat: 150000,
  evaluation: 900000
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

export function reloadSkills() {
  return requestJson('/skills/reload', { method: 'POST', timeoutMs: TIMEOUT.write })
}

export function requestKnowledgeStats(signal) {
  return requestJson('/knowledge/stats', { signal })
}

export function runEvaluation(body = null) {
  return requestJson('/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    timeoutMs: TIMEOUT.evaluation
  }).then(normalizeEvaluationResponse)
}

export function requestSearch(query, topK = 5, signal) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
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
      user_id: settings.userId || 'anonymous',
      conv_id: settings.conversationId || undefined
    }),
    signal,
    timeoutMs: TIMEOUT.chat
  }).then(normalizeChatResponse)
}

export function requestToolTrace(requestId, signal) {
  if (!requestId) return Promise.resolve(null)
  return requestJson(`/trace/tool/${encodeURIComponent(requestId)}`, { signal }).then(normalizeToolTraceResponse)
}

export function requestRecentTraces(limit = 20, signal) {
  return requestJson(`/trace/recent?limit=${limit}`, { signal })
}

export function requestTraceTree(traceId, signal) {
  return requestJson(`/trace/${encodeURIComponent(traceId)}`, { signal })
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

function normalizeChatResponse(raw) {
  const degradations = normalizeDegradations(raw.degradations)
  return {
    conversationId: raw.conv_id || '',
    requestId: raw.request_id || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    agentType: raw.agent_type || '',
    agentTypes: raw.agent_types || [],
    primaryAgent: raw.primary_agent || '',
    supportingAgents: raw.supporting_agents || [],
    toolsUsed: raw.tools_used || [],
    routingReason: raw.routing_reason || '',
    routingConfidence: Number(raw.routing_confidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? 0),
    intentSourceScores: raw.intent_source_scores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used),
    degradations,
    degraded: Boolean(raw.degraded),
    raw
  }
}

function normalizeSearchResponse(raw) {
  return {
    query: raw.query || '',
    results: raw.results || [],
    reranked: Boolean(raw.reranked),
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

function normalizeEvaluationResponse(raw) {
  return {
    passRate: Number(raw?.pass_rate ?? 0),
    total: Number(raw?.total ?? 0),
    passed: Number(raw?.passed ?? 0),
    avgScores: raw?.avg_scores || {},
    regressions: raw?.regressions || [],
    recommendations: raw?.recommendations || [],
    results: (raw?.results || []).map(item => ({
      testId: item?.test_id || '',
      passed: Boolean(item?.passed),
      scores: item?.scores || {},
      detail: item?.detail || '',
      metadata: item?.metadata || {}
    }))
  }
}

function normalizeDegradations(events) {
  return (events || []).map(event => ({
    source: event?.source || '',
    code: event?.code || '',
    message: event?.message || ''
  }))
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || [],
      toolCalls: trace.tool_calls || []
    },
    raw
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
