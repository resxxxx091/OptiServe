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
const kind = computed(() => String(props.span.name || '').split(/[:.]/)[0].replace(/[^a-zA-Z0-9_-]/g, '') || 'span')
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
  <details class="span-node" :class="[`span-kind-${kind}`, { leaf: !hasDetail }]" :open="depth <= 2">
    <summary class="span-row">
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
