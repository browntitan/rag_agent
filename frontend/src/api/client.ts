import type { BackendStatus, ProgressEvent, UploadResult } from '../types'

const API_BASE = '/v1'
const BACKEND_API_URL = 'http://localhost:8000'
const BACKEND_START_COMMAND = 'python run.py serve-api --host 0.0.0.0 --port 8000'

export const BACKEND_UNAVAILABLE_MESSAGE =
  `The backend API is unavailable at ${BACKEND_API_URL}. Start it with "${BACKEND_START_COMMAND}" and then retry.`

class ApiError extends Error {
  status?: number
  backendUnavailable: boolean

  constructor(message: string, options?: { status?: number; backendUnavailable?: boolean }) {
    super(message)
    this.name = 'ApiError'
    this.status = options?.status
    this.backendUnavailable = options?.backendUnavailable ?? false
  }
}

function createBackendUnavailableError(): ApiError {
  return new ApiError(BACKEND_UNAVAILABLE_MESSAGE, { backendUnavailable: true })
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload === 'string') {
    const value = payload.trim()
    return value.length > 0 ? value : null
  }

  if (!payload || typeof payload !== 'object') return null

  const record = payload as Record<string, unknown>

  if (typeof record.detail === 'string' && record.detail.trim().length > 0) {
    return record.detail.trim()
  }

  if (Array.isArray(record.detail)) {
    const parts = record.detail
      .map(item => extractErrorMessage(item))
      .filter((item): item is string => !!item)
    if (parts.length > 0) return parts.join(', ')
  }

  if (typeof record.error === 'string' && record.error.trim().length > 0) {
    return record.error.trim()
  }

  if (typeof record.message === 'string' && record.message.trim().length > 0) {
    return record.message.trim()
  }

  return null
}

async function readErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get('content-type') ?? ''

  if (contentType.includes('application/json')) {
    try {
      const payload = await response.clone().json()
      const message = extractErrorMessage(payload)
      if (message) return message
    } catch {
      // Fall back to raw text when the body is not valid JSON.
    }
  }

  try {
    return (await response.text()).trim()
  } catch {
    return ''
  }
}

async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  let response: Response

  try {
    response = await fetch(path, init)
  } catch (error) {
    if (isAbortError(error)) throw error
    throw createBackendUnavailableError()
  }

  if (!response.ok) {
    const message = await readErrorMessage(response)

    if (response.status === 500 && message.length === 0) {
      throw createBackendUnavailableError()
    }

    throw new ApiError(message || `HTTP ${response.status}`, { status: response.status })
  }

  return response
}

export function isBackendUnavailableError(error: unknown): boolean {
  return error instanceof ApiError && error.backendUnavailable
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) return error.message
  return 'Unknown error'
}

export async function getModelId(): Promise<string> {
  try {
    const res = await apiFetch(`${API_BASE}/models`)
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

  const res = await apiFetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Conversation-ID': conversationId,
    },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  })

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

  const res = await apiFetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: { 'X-Conversation-ID': conversationId },
    body: form,
  })

  return res.json() as Promise<UploadResult>
}

/** Check if the backend is healthy and ready. */
export async function checkHealth(): Promise<BackendStatus> {
  try {
    await apiFetch('/health/ready')
    return { ready: true }
  } catch (error) {
    return { ready: false, message: getErrorMessage(error) }
  }
}
