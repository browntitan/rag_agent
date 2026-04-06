import { useCallback, useEffect, useState } from 'react'
import { checkHealth } from './api/client'
import { useChat } from './hooks/useChat'
import { useFileUpload } from './hooks/useFileUpload'
import { ChatWindow } from './components/ChatWindow'
import { ChatInput } from './components/ChatInput'
import { FileUpload } from './components/FileUpload'
import type { BackendStatus } from './types'
import './App.css'

export default function App() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({ ready: true })
  const [isCheckingBackend, setIsCheckingBackend] = useState(false)

  const refreshBackendStatus = useCallback(async () => {
    setIsCheckingBackend(true)
    const status = await checkHealth()
    setBackendStatus(status)
    setIsCheckingBackend(false)
  }, [])

  useEffect(() => {
    void refreshBackendStatus()
  }, [refreshBackendStatus])

  const handleBackendUnavailable = useCallback((message: string) => {
    setBackendStatus({ ready: false, message })
  }, [])

  const { messages, isLoading, conversationId, sendMessage, stopGeneration, newChat } =
    useChat(handleBackendUnavailable)
  const { isUploading, uploadedFiles, lastError, upload } =
    useFileUpload(conversationId, handleBackendUnavailable)

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__title">Agentic RAG Chatbot</span>
        <div className="app__actions">
          <FileUpload
            isUploading={isUploading}
            uploadedFiles={uploadedFiles}
            lastError={lastError}
            disabled={!backendStatus.ready}
            onUpload={upload}
          />
          <button className="app__new-chat-btn" onClick={newChat}>
            New Chat
          </button>
        </div>
      </header>
      {!backendStatus.ready && (
        <div className="app__status-banner" role="alert">
          <div className="app__status-copy">
            <span className="app__status-title">Backend unavailable</span>
            <span>{backendStatus.message}</span>
          </div>
          <button
            className="app__status-btn"
            onClick={() => void refreshBackendStatus()}
            disabled={isCheckingBackend}
          >
            {isCheckingBackend ? 'Checking...' : 'Retry'}
          </button>
        </div>
      )}
      <ChatWindow messages={messages} />
      <ChatInput
        onSend={sendMessage}
        onStop={stopGeneration}
        isLoading={isLoading}
        disabled={isUploading || !backendStatus.ready}
      />
    </div>
  )
}
