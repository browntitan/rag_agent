import { useChat } from './hooks/useChat'
import { useFileUpload } from './hooks/useFileUpload'
import { ChatWindow } from './components/ChatWindow'
import { ChatInput } from './components/ChatInput'
import { FileUpload } from './components/FileUpload'
import './App.css'

export default function App() {
  const { messages, isLoading, conversationId, sendMessage, stopGeneration, newChat } = useChat()
  const { isUploading, uploadedFiles, lastError, upload } = useFileUpload(conversationId)

  return (
    <div className="app">
      <header className="app__header">
        <span className="app__title">Agentic RAG Chatbot</span>
        <div className="app__actions">
          <FileUpload
            isUploading={isUploading}
            uploadedFiles={uploadedFiles}
            lastError={lastError}
            onUpload={upload}
          />
          <button className="app__new-chat-btn" onClick={newChat}>
            New Chat
          </button>
        </div>
      </header>
      <ChatWindow messages={messages} />
      <ChatInput
        onSend={sendMessage}
        onStop={stopGeneration}
        isLoading={isLoading}
        disabled={isUploading}
      />
    </div>
  )
}
