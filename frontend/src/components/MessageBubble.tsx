import type { Message } from '../types'
import './MessageBubble.css'

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function MessageBubble({ message }: { message: Message }) {
  return (
    <div className={`message-bubble message-bubble--${message.role}`}>
      <div>
        <div className="message-bubble__content">
          {message.isStreaming && !message.content ? (
            <span className="typing-indicator">
              <span /><span /><span />
            </span>
          ) : (
            message.content
          )}
        </div>
        <div className="message-bubble__time">{formatTime(message.timestamp)}</div>
      </div>
    </div>
  )
}
