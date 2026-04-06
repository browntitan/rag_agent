export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  isStreaming?: boolean
  steps?: ProgressEvent[]   // agent steps accumulated during streaming
}

export interface BackendStatus {
  ready: boolean
  message?: string
}

export interface UploadResult {
  object: string
  tenant_id: string
  ingested_count: number
  doc_ids: string[]
  filenames: string[]
  errors: string[]
  workspace_copies?: string[]
}

export type ProgressEventType =
  | 'agent_start'
  | 'tool_call'
  | 'tool_result'
  | 'tool_error'

export interface ProgressEvent {
  type: ProgressEventType
  id?: string
  // agent_start fields
  node?: string
  label?: string
  // tool_call fields
  tool?: string
  input?: unknown
  // tool_result / tool_error fields
  output?: unknown
  error?: string
  duration_ms?: number
  timestamp: number
}
