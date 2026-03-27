export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  isStreaming?: boolean
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
