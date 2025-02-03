import { createOpenAI } from "@ai-sdk/openai";
import { streamText, smoothStream } from "ai";

export const maxDuration = 30;

const promptTemplates = {
  rewrite: (text: string, context: string) => `
Rewrite the following text in a different way while maintaining its meaning and technical accuracy.
Keep the same level of detail but use different wording and structure.

Original text in its context:
${context}

Text to rewrite:
${text}

Rewritten version:`,

  describe: (text: string, context: string) => `
Provide a detailed explanation and additional context for the following text.
Include relevant examples, use cases, and technical details that would help someone better understand this concept.

Text in context:
${context}

Text to explain:
${text}

Detailed explanation:`,

  example: (text: string, context: string) => `
Generate practical, real-world examples that illustrate the concept described in the following text.
Include code examples where relevant and explain how they relate to the concept.

Concept in context:
${context}

Specific concept to exemplify:
${text}

Examples:`,
};

export async function POST(req: Request) {
  try {
    const { option, selectedText, markdownContent } = await req.json();

    if (!option || !selectedText || !markdownContent) {
      return new Response("Missing required fields", { status: 400 });
    }

    const openai = createOpenAI({
      baseURL: process.env.OPENAI_BASE_URL,
      apiKey: process.env.OPENAI_API_KEY,
    });

    const promptTemplate =
      promptTemplates[option as keyof typeof promptTemplates];
    if (!promptTemplate) {
      return new Response("Invalid option", { status: 400 });
    }

    const prompt = promptTemplate(selectedText, markdownContent);

    const result = streamText({
      model: openai("gpt-4o-mini"),
      prompt,
      experimental_transform: [smoothStream()],
    });

    return result.toDataStreamResponse();
  } catch (error) {
    console.error("Stream error:", error);
    return new Response("Internal server error", { status: 500 });
  }
}
