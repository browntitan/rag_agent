import { useState, useRef, useCallback } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { getErrorMessage, isBackendUnavailableError, streamChatCompletion } from '../api/client'
import type { Message } from '../types'

export function useChat(onBackendUnavailable?: (message: string) => void) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId] = useState<string>(() => {
    const stored = localStorage.getItem('conversationId')
    if (stored) return stored
    const id = uuidv4()
    localStorage.setItem('conversationId', id)
    return id
  })
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isLoading) return

      const userMsg: Message = {
        id: uuidv4(),
        role: 'user',
        content: content.trim(),
        timestamp: Date.now(),
      }

      const assistantId = uuidv4()
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        isStreaming: true,
        steps: [],
      }

      setMessages(prev => [...prev, userMsg, assistantMsg])
      setIsLoading(true)

      const controller = new AbortController()
      abortRef.current = controller

      // Build message history for the API (exclude the placeholder assistant msg)
      const history = [...messages, userMsg].map(m => ({
        role: m.role,
        content: m.content,
      }))

      try {
        const stream = streamChatCompletion(history, conversationId, controller.signal)

        for await (const event of stream) {
          if (event.kind === 'content') {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + event.text }
                  : m,
              ),
            )
          } else if (event.kind === 'progress') {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, steps: [...(m.steps ?? []), event.event] }
                  : m,
              ),
            )
          } else if (event.kind === 'done') {
            break
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: m.content || '[Stopped]', isStreaming: false }
                : m,
            ),
          )
        } else {
          const errMsg = getErrorMessage(err)
          if (isBackendUnavailableError(err)) {
            onBackendUnavailable?.(errMsg)
          }
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId
                ? { ...m, content: errMsg, isStreaming: false }
                : m,
            ),
          )
        }
      } finally {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        )
        setIsLoading(false)
        abortRef.current = null
      }
    },
    [messages, isLoading, conversationId, onBackendUnavailable],
  )

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const newChat = useCallback(() => {
    const newId = uuidv4()
    localStorage.setItem('conversationId', newId)
    window.location.reload()
  }, [])

  return { messages, isLoading, conversationId, sendMessage, stopGeneration, newChat }
}
