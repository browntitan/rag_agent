import { useState, useCallback, type KeyboardEvent } from 'react'
import './ChatInput.css'

interface Props {
  onSend: (content: string) => void
  onStop: () => void
  isLoading: boolean
  disabled?: boolean
}

export function ChatInput({ onSend, onStop, isLoading, disabled }: Props) {
  const [input, setInput] = useState('')

  const handleSend = useCallback(() => {
    const trimmed = input.trim()
    if (!trimmed || isLoading || disabled) return
    onSend(trimmed)
    setInput('')
  }, [input, isLoading, disabled, onSend])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-input">
      <textarea
        className="chat-input__textarea"
        rows={1}
        placeholder="Type a message..."
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      {isLoading ? (
        <button className="chat-input__btn chat-input__btn--stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button
          className="chat-input__btn chat-input__btn--send"
          onClick={handleSend}
          disabled={!input.trim() || disabled}
        >
          Send
        </button>
      )}
    </div>
  )
}
