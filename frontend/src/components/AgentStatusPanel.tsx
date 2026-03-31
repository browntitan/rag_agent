import { useState, useEffect, useRef } from 'react'
import type { ProgressEvent } from '../types'
import './AgentStatusPanel.css'

interface Props {
  steps: ProgressEvent[]
  isStreaming: boolean
}

// Node → icon mapping
const NODE_ICONS: Record<string, string> = {
  supervisor: '🔀',
  rag_agent: '🔍',
  utility_agent: '🔧',
  parallel_planner: '📋',
  rag_worker: '🔍',
  rag_synthesizer: '✨',
  data_analyst: '📊',
  clarify: '❓',
  evaluator: '✅',
}

const TOOL_ICONS: Record<string, string> = {
  search_document: '📄',
  search_all_documents: '🗂️',
  full_text_search_document: '🔡',
  resolve_document: '🔎',
  list_document_structure: '📑',
  extract_clauses: '📜',
  chunk_expander: '↔️',
  citation_validator: '📌',
  query_rewriter: '✏️',
  web_search_fallback: '🌐',
  graph_search_local: '🕸️',
  graph_search_global: '🌐',
  calculator: '🧮',
  list_indexed_docs: '📋',
  memory_save: '💾',
  memory_load: '📥',
  scratchpad_write: '📝',
  scratchpad_read: '📖',
  code_interpreter: '💻',
}

function formatDuration(ms?: number): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function truncate(s: string, max = 120): string {
  if (s.length <= max) return s
  return s.slice(0, max) + '…'
}

function renderValue(val: unknown): string {
  if (val == null) return ''
  if (typeof val === 'string') return val
  try {
    return JSON.stringify(val, null, 2)
  } catch {
    return String(val)
  }
}

interface ToolCallItemProps {
  callEvent: ProgressEvent
  resultEvent?: ProgressEvent
}

function ToolCallItem({ callEvent, resultEvent }: ToolCallItemProps) {
  const [expanded, setExpanded] = useState(false)
  const icon = TOOL_ICONS[callEvent.tool ?? ''] ?? '🔧'
  const hasError = resultEvent?.type === 'tool_error'

  return (
    <div className={`tool-call-item ${hasError ? 'tool-call-error' : ''}`}>
      <button className="tool-call-header" onClick={() => setExpanded(e => !e)}>
        <span className="tool-icon">{icon}</span>
        <span className="tool-name">{callEvent.tool}</span>
        {resultEvent && (
          <span className={`tool-badge ${hasError ? 'badge-error' : 'badge-ok'}`}>
            {hasError ? 'error' : formatDuration(resultEvent.duration_ms)}
          </span>
        )}
        {!resultEvent && <span className="tool-badge badge-running">running…</span>}
        <span className="tool-chevron">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && (
        <div className="tool-call-body">
          {callEvent.input != null && (
            <div className="tool-section">
              <div className="tool-section-label">Input</div>
              <pre className="tool-code">{renderValue(callEvent.input)}</pre>
            </div>
          )}
          {resultEvent && (
            <div className="tool-section">
              <div className="tool-section-label">{hasError ? 'Error' : 'Output'}</div>
              <pre className="tool-code">
                {hasError
                  ? (resultEvent.error ?? 'Unknown error')
                  : truncate(renderValue(resultEvent.output), 600)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function AgentStatusPanel({ steps, isStreaming }: Props) {
  const [collapsed, setCollapsed] = useState(false)
  const prevStreamingRef = useRef(isStreaming)

  // Auto-collapse when streaming finishes
  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      setCollapsed(true)
    }
    prevStreamingRef.current = isStreaming
  }, [isStreaming])

  if (!steps || steps.length === 0) {
    if (!isStreaming) return null
    // Show a subtle placeholder while waiting for first event
    return (
      <div className="agent-status-panel agent-status-waiting">
        <span className="agent-dot" />
        <span className="agent-waiting-text">Agent is thinking…</span>
      </div>
    )
  }

  // Get the most recent agent_start for the live status label
  const lastAgentStart = [...steps].reverse().find(s => s.type === 'agent_start')
  const currentLabel = lastAgentStart?.label ?? 'Processing…'
  const currentIcon = NODE_ICONS[lastAgentStart?.node ?? ''] ?? '⚙️'

  // Count tool calls for summary
  const toolCallCount = steps.filter(s => s.type === 'tool_call').length

  // Build paired tool call → result list
  const toolPairs: Array<{ call: ProgressEvent; result?: ProgressEvent }> = []
  for (const step of steps) {
    if (step.type === 'tool_call') {
      const result = steps.find(
        s => (s.type === 'tool_result' || s.type === 'tool_error') && s.id === step.id,
      )
      toolPairs.push({ call: step, result })
    }
  }

  return (
    <div className="agent-status-panel">
      <button className="agent-status-header" onClick={() => setCollapsed(c => !c)}>
        <span className="agent-header-left">
          {isStreaming ? (
            <>
              <span className="agent-dot agent-dot-live" />
              <span className="agent-current-label">
                {currentIcon} {currentLabel}
              </span>
            </>
          ) : (
            <>
              <span className="agent-done-icon">✓</span>
              <span className="agent-summary-label">
                {toolCallCount > 0
                  ? `${toolCallCount} tool call${toolCallCount !== 1 ? 's' : ''}`
                  : 'Completed'}
              </span>
            </>
          )}
        </span>
        <span className="agent-chevron">{collapsed ? '▶' : '▼'}</span>
      </button>

      {!collapsed && (
        <div className="agent-status-body">
          {/* Agent nodes visited */}
          <div className="agent-nodes-row">
            {steps
              .filter(s => s.type === 'agent_start')
              .map((s, i) => (
                <span key={i} className="agent-node-chip">
                  {NODE_ICONS[s.node ?? ''] ?? '⚙️'} {s.label}
                </span>
              ))}
          </div>

          {/* Tool calls */}
          {toolPairs.length > 0 && (
            <div className="tool-calls-list">
              {toolPairs.map(({ call, result }, i) => (
                <ToolCallItem key={i} callEvent={call} resultEvent={result} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
