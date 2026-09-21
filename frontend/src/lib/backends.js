export const API_BASE = String(import.meta.env.VITE_PYTHON_API_URL || '/api/python').replace(/\/+$/, '')

const SETTINGS_KEY = 'optiserve.frontend.settings'

export function createInitialSettings() {
  const saved = readSettings()
  return {
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || ''
  }
}

export function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
}

export function requestHealth() {
  return requestJson('/health')
}

export function requestMonitor() {
  return requestJson('/monitor')
}

export function requestSkills() {
  return requestJson('/skills')
}

export function reloadSkills() {
  return requestJson('/skills/reload', { method: 'POST' })
}

export function requestKnowledgeStats() {
  return requestJson('/knowledge/stats')
}

export function runEvaluation(body = null) {
  return requestJson('/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
}

export function requestSearch(query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestJson(`/search?${params}`, { method: 'POST' })
}

export function requestChat(settings, message) {
  return requestJson('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      user_id: settings.userId || 'anonymous',
      conv_id: settings.conversationId || undefined
    })
  }).then(normalizeChatResponse)
}

export function requestToolTrace(requestId) {
  if (!requestId) return Promise.resolve(null)
  return requestJson(`/trace/tool/${encodeURIComponent(requestId)}`).then(normalizeToolTraceResponse)
}

export function requestRecentTraces(limit = 20) {
  return requestJson(`/trace/recent?limit=${limit}`)
}

export function requestTraceTree(traceId) {
  return requestJson(`/trace/${encodeURIComponent(traceId)}`)
}

export function addKnowledge(documents) {
  return requestJson('/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export function uploadKnowledge(file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson('/knowledge/upload', { method: 'POST', body: form })
}

function normalizeChatResponse(raw) {
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
    raw
  }
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
  const response = await fetch(`${API_BASE}${path}`, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')
  } catch {
    return {}
  }
}
