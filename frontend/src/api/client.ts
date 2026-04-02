import type { ProgressEvent, UploadResult } from '../types'

const API_BASE = '/v1'

export async function getModelId(): Promise<string> {
  try {
    const res = await fetch(`${API_BASE}/models`)
    if (!res.ok) return 'enterprise-agent'
    const data = await res.json()
    return data?.data?.[0]?.id ?? 'enterprise-agent'
  } catch {
    return 'enterprise-agent'
  }
}

let _cachedModelId: string | null = null

async function modelId(): Promise<string> {
  if (!_cachedModelId) _cachedModelId = await getModelId()
  return _cachedModelId
}

export type StreamEvent =
  | { kind: 'content'; text: string }
  | { kind: 'progress'; event: ProgressEvent }
  | { kind: 'done' }

/**
 * Streams a chat completion, yielding both content tokens and progress events.
 *
 * Named SSE events (`event: progress`) are yielded as StreamEvent { kind: 'progress' }.
 * Content tokens from chat.completion.chunk payloads are yielded as { kind: 'content' }.
 */
export async function* streamChatCompletion(
  messages: Array<{ role: string; content: string }>,
  conversationId: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const model = await modelId()

  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Conversation-ID': conversationId,
    },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  })

  if (!res.ok) {
    const errorText = await res.text().catch(() => 'Unknown error')
    throw new Error(`HTTP ${res.status}: ${errorText}`)
  }

  if (!res.body) throw new Error('No response body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // SSE messages are separated by double newlines
      const messages_raw = buffer.split(/\n\n/)
      // Keep the last (possibly incomplete) chunk in the buffer
      buffer = messages_raw.pop() ?? ''

      for (const msg of messages_raw) {
        if (!msg.trim()) continue

        // Parse SSE message: may contain "event:" and "data:" lines
        const lines = msg.split('\n')
        let eventType = ''
        let dataLine = ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            dataLine = line.slice(5).trim()
          }
        }

        if (!dataLine) continue
        if (dataLine === '[DONE]') {
          yield { kind: 'done' }
          return
        }

        try {
          const parsed = JSON.parse(dataLine)

          if (eventType === 'progress') {
            // Named progress event from the backend callback
            yield { kind: 'progress', event: parsed as ProgressEvent }
          } else {
            // Standard OpenAI chat.completion.chunk
            const content = parsed?.choices?.[0]?.delta?.content
            if (typeof content === 'string' && content.length > 0) {
              yield { kind: 'content', text: content }
            }
          }
        } catch {
          // Skip malformed JSON lines
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/** Upload files to the backend for ingestion. */
export async function uploadFiles(
  files: File[],
  conversationId: string,
): Promise<UploadResult> {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: { 'X-Conversation-ID': conversationId },
    body: form,
  })

  if (!res.ok) {
    const msg = await res.text().catch(() => 'Upload failed')
    throw new Error(msg)
  }
  return res.json() as Promise<UploadResult>
}

/** Check if the backend is healthy and ready. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch('/health/ready')
    return res.ok
  } catch {
    return false
  }
}
