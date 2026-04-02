import { convertToModelMessages, generateId, streamText, type UIMessage } from 'ai';

import {
  AGENTIC_GATEWAY_MODEL,
  createAgenticChatbotProvider,
} from '@/lib/agentic-provider';

export const maxDuration = 60;

type ChatRequest = {
  id?: string;
  messages: UIMessage[];
  forceAgent?: boolean;
};

export async function POST(req: Request) {
  const { id, messages, forceAgent = false } = (await req.json()) as ChatRequest;
  const conversationId = id ?? generateId();
  const provider = createAgenticChatbotProvider(conversationId);

  const result = streamText({
    model: provider.chatModel(AGENTIC_GATEWAY_MODEL),
    messages: convertToModelMessages(messages),
    providerOptions: {
      agenticChatbot: {
        metadata: {
          force_agent: forceAgent,
        },
      },
    },
  });

  return result.toUIMessageStreamResponse({
    headers: {
      'X-Conversation-ID': conversationId,
    },
  });
}
