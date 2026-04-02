import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

export const AGENTIC_GATEWAY_MODEL =
  process.env.AGENTIC_GATEWAY_MODEL ?? 'enterprise-agent';

export function createAgenticChatbotProvider(conversationId?: string) {
  return createOpenAICompatible({
    name: 'agentic-chatbot',
    apiKey: process.env.AGENTIC_GATEWAY_API_KEY ?? 'dev-key',
    baseURL: process.env.AGENTIC_GATEWAY_BASE_URL ?? 'http://localhost:8000/v1',
    headers: conversationId
      ? {
          'X-Conversation-ID': conversationId,
        }
      : undefined,
    includeUsage: true,
  });
}
