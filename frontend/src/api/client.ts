import type { UploadResult } from '../types'

const API_BASE = '/v1'

/** Discover the model ID from the backend, with a sensible fallback. */
async function getModelId(): Promise<string> {
  try {
    const resp = await fetch(`${API_BASE}/models`)
    if (resp.ok) {
      const body = await resp.json()
      const id = body?.data?.[0]?.id
      if (typeof id === 'string') return id
    }
  } catch {
    // fall through
  }
  return 'enterprise-agent'
}

let _cachedModelId: string | null = null

async function modelId(): Promise<string> {
  if (!_cachedModelId) _cachedModelId = await getModelId()
  return _cachedModelId
}

/**
 * Stream chat completions from the backend.
 * Yields content tokens as they arrive via SSE.
 */
export async function* streamChatCompletion(
  messages: { role: string; content: string }[],
  conversationId: string,
  abortSignal?: AbortSignal,
): AsyncGenerator<string> {
  const model = await modelId()

  const response = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Conversation-ID': conversationId,
    },
    body: JSON.stringify({ model, messages, stream: true }),
    signal: abortSignal,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`API error ${response.status}: ${text}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data: ')) continue
      const data = trimmed.slice(6)
      if (data === '[DONE]') return

      try {
        const parsed = JSON.parse(data)
        const content = parsed.choices?.[0]?.delta?.content
        if (content) yield content
      } catch {
        // skip malformed chunks
      }
    }
  }
}

/** Upload files to the backend for ingestion. */
export async function uploadFiles(
  files: File[],
  conversationId: string,
): Promise<UploadResult> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: { 'X-Conversation-ID': conversationId },
    body: formData,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText)
    throw new Error(`Upload error ${response.status}: ${text}`)
  }

  return response.json()
}

/** Check if the backend is healthy and ready. */
export async function checkHealth(): Promise<boolean> {
  try {
    const resp = await fetch('/health/ready')
    return resp.ok
  } catch {
    return false
  }
}
