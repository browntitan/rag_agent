'use client';

import { useState } from 'react';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

export default function Page() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: '/api/chat',
    }),
  });

  return (
    <main style={{ margin: '0 auto', maxWidth: 720, padding: '3rem 1.5rem' }}>
      <h1>Main Agentic Runtime</h1>
      <p>UI layer in Next.js, agentic execution in the Python gateway.</p>

      <div style={{ display: 'grid', gap: '1rem', marginTop: '2rem' }}>
        {messages.map(message => (
          <section
            key={message.id}
            style={{
              border: '1px solid #d4d4d8',
              borderRadius: 12,
              padding: '1rem',
              background: message.role === 'user' ? '#fafaf9' : '#ffffff',
            }}
          >
            <strong style={{ display: 'block', marginBottom: '.5rem' }}>
              {message.role === 'user' ? 'You' : 'Assistant'}
            </strong>
            {message.parts.map((part, index) =>
              part.type === 'text' ? <p key={index}>{part.text}</p> : null,
            )}
          </section>
        ))}
      </div>

      <form
        onSubmit={event => {
          event.preventDefault();
          const text = input.trim();
          if (!text) {
            return;
          }
          sendMessage({ text });
          setInput('');
        }}
        style={{ display: 'grid', gap: '.75rem', marginTop: '2rem' }}
      >
        <textarea
          value={input}
          onChange={event => setInput(event.target.value)}
          placeholder="Ask the Python agentic backend a question..."
          rows={5}
          style={{ borderRadius: 12, border: '1px solid #a1a1aa', padding: '1rem' }}
        />
        <button
          type="submit"
          disabled={status === 'submitted' || status === 'streaming'}
          style={{
            width: 'fit-content',
            border: 0,
            borderRadius: 999,
            background: '#18181b',
            color: '#fafafa',
            padding: '.8rem 1.2rem',
          }}
        >
          {status === 'submitted' || status === 'streaming' ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </main>
  );
}
