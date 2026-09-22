<script setup>
import { computed } from 'vue'

const props = defineProps({
  span: { type: Object, required: true },
  depth: { type: Number, default: 0 },
  rootLatency: { type: Number, default: 0 }
})

// span 树来自后端 GET /trace/{trace_id}，字段是 snake_case 递归结构，这里不做改名
const children = computed(() => props.span.children || [])
const attrs = computed(() => props.span.attrs || {})
const inputAttr = computed(() => attrs.value.input)
const metaAttrs = computed(() =>
  Object.entries(attrs.value).filter(([key]) => key !== 'input')
)

// 后端 span 名前缀不统一：agent:x / rag.x / tool:x 带分隔符，llm_round_1、memory_read 是纯下划线。
// 只按 [:.] 切分会让 llm_round_N 各成一类、配色规则失效，所以这里归一到层级组。
const KIND_GROUPS = {
  agent: 'agent',
  llm: 'llm',
  tool: 'tool',
  rag: 'rag',
  memory: 'memory',
  chat: 'orchestrator',
  search: 'orchestrator',
  route: 'orchestrator',
  compose: 'orchestrator',
  intent: 'orchestrator',
  request: 'orchestrator'
}

const rawKind = computed(() => String(props.span.name || '').split(/[:._]/)[0].toLowerCase())
const kindGroup = computed(() => KIND_GROUPS[rawKind.value] || 'orchestrator')
const hasDetail = computed(
  () => children.value.length > 0 || metaAttrs.value.length > 0 || inputAttr.value !== undefined || Boolean(props.span.error)
)
const barWidth = computed(() => {
  const total = Number(props.rootLatency) || 0
  if (total <= 0) return 0
  return Math.min(100, (Number(props.span.latency_ms) || 0) / total * 100)
})

function attrText(value) {
  if (Array.isArray(value)) return value.join(' · ')
  if (value === null || value === undefined) return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<template>
  <details class="span-node" :class="[`span-kind-${kindGroup}`, { leaf: !hasDetail }]" :open="depth <= 2">
    <summary class="span-row">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" aria-hidden="true">
        <path d="M6 3.5 10.5 8 6 12.5" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
      <span class="span-tick" :title="rawKind" aria-hidden="true"></span>
      <span class="span-name">{{ span.name }}</span>
      <span class="span-bar"><i :style="{ width: `${barWidth}%` }"></i></span>
      <span class="span-latency">{{ span.latency_ms ?? 0 }} ms</span>
      <span v-if="span.status === 'error'" class="span-badge danger">error</span>
      <span v-if="span.events?.length" class="span-badge">{{ span.events.length }} events</span>
    </summary>

    <div class="span-detail">
      <p v-if="span.error" class="span-error">{{ span.error }}</p>
      <div v-if="metaAttrs.length" class="span-attrs">
        <span v-for="[key, value] in metaAttrs" :key="key" class="span-attr">
          <em>{{ key }}</em>{{ attrText(value) }}
        </span>
      </div>
      <pre v-if="inputAttr !== undefined" class="span-input">{{ JSON.stringify(inputAttr ?? {}, null, 2) }}</pre>
      <SpanNode
        v-for="child in children"
        :key="child.span_id"
        :span="child"
        :depth="depth + 1"
        :root-latency="rootLatency"
      />
    </div>
  </details>
</template>
