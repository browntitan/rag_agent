import './MessageBubble.css'
import { AgentStatusPanel } from './AgentStatusPanel'
import type { Message } from '../types'

interface Props {
  message: Message
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const hasSteps = message.steps && message.steps.length > 0
  const showPanel = !isUser && (hasSteps || message.isStreaming)

  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <div>
        {/* Agent status / tool call panel for assistant messages */}
        {showPanel && (
          <AgentStatusPanel
            steps={message.steps ?? []}
            isStreaming={!!message.isStreaming}
          />
        )}

        {/* Message content */}
        <div className="message-bubble__content">
          {message.content ? (
            <span className="message-text">{message.content}</span>
          ) : message.isStreaming ? (
            <span className="typing-indicator">
              <span /><span /><span />
            </span>
          ) : (
            <span className="message-text" />
          )}
        </div>

        <div className="message-bubble__time">{formatTime(message.timestamp)}</div>
      </div>
    </div>
  )
}
