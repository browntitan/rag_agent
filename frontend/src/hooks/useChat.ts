import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Message } from '../types'
import { streamChatCompletion } from '../api/client'

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState(() => {
    return localStorage.getItem('conversationId') || uuidv4()
  })
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    localStorage.setItem('conversationId', conversationId)

    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: content.trim(),
      timestamp: Date.now(),
    }

    const assistantId = uuidv4()

    setMessages(prev => [
      ...prev,
      userMsg,
      { id: assistantId, role: 'assistant', content: '', timestamp: Date.now(), isStreaming: true },
    ])
    setIsLoading(true)

    // Build API messages from full history + new user message
    const apiMessages = [...messages, userMsg].map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      abortRef.current = new AbortController()

      for await (const chunk of streamChatCompletion(
        apiMessages,
        conversationId,
        abortRef.current.signal,
      )) {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: m.content + chunk }
              : m,
          ),
        )
      }

      // Mark streaming complete
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId ? { ...m, isStreaming: false } : m,
        ),
      )
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: `Error: ${(error as Error).message}`, isStreaming: false }
              : m,
          ),
        )
      } else {
        // On abort, keep whatever content we have and stop streaming
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        )
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }, [messages, isLoading, conversationId])

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
    setIsLoading(false)
  }, [])

  const newChat = useCallback(() => {
    const newId = uuidv4()
    setConversationId(newId)
    setMessages([])
    localStorage.setItem('conversationId', newId)
  }, [])

  return { messages, isLoading, conversationId, sendMessage, stopGeneration, newChat }
}
