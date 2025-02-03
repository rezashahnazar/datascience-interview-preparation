"use client";

import { Drawer } from "vaul";
import { useAIResponse } from "@/lib/hooks/use-ai-response";
import { Loader2, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { MagicalLoadingOverlay } from "./MagicalLoadingOverlay";
import { useScrollControl } from "@/lib/hooks/use-scroll-control";
import { ScrollToBottomButton } from "./ScrollToBottomButton";

interface AIResponseDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  option: string;
  selectedText: string;
  markdownContent: string;
}

interface CodeProps extends React.ClassAttributes<HTMLElement> {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export default function AIResponseDrawer({
  isOpen,
  onClose,
  option,
  selectedText,
  markdownContent,
}: AIResponseDrawerProps) {
  const { output, error, isLoading, handleOpenChange } = useAIResponse({
    isOpen,
    onClose,
    option,
    selectedText,
    markdownContent,
  });

  const { containerRef, contentRef, showScrollButton, scrollToBottom } =
    useScrollControl(isLoading);

  return (
    <Drawer.Root open={isOpen} onOpenChange={handleOpenChange}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 bg-black/40" />
        <Drawer.Content
          aria-describedby={undefined}
          className="fixed bottom-0 left-0 right-0 mt-24 flex h-[80dvh] flex-col rounded-t-[10px] bg-white dark:bg-zinc-900"
        >
          <Drawer.Title className="sr-only">AI Response</Drawer.Title>
          <div className="absolute inset-0 z-50 pointer-events-none">
            <MagicalLoadingOverlay isLoading={isLoading} />
          </div>
          <div className="relative flex-1">
            <div
              ref={containerRef}
              className="absolute inset-0 overflow-y-auto rounded-t-[10px] bg-zinc-100 dark:bg-zinc-800 p-4 scroll-smooth"
            >
              <div className="mx-auto max-w-2xl pb-16">
                <div className="mb-6 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-indigo-500" />
                    <h2 className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-xl font-bold text-transparent">
                      Magical Assistant
                    </h2>
                  </div>
                  {isLoading && (
                    <div className="flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Crafting magic...
                    </div>
                  )}
                </div>

                {error && (
                  <div className="mb-4 rounded-md bg-red-50 dark:bg-red-900/20 p-4 text-sm text-red-500 dark:text-red-400">
                    {error}
                  </div>
                )}

                <div
                  ref={contentRef}
                  className="prose prose-sm dark:prose-invert max-w-none"
                >
                  <ReactMarkdown
                    rehypePlugins={[rehypeHighlight, rehypeRaw]}
                    remarkPlugins={[remarkGfm]}
                    components={{
                      pre: ({ ...props }) => (
                        <pre
                          className="bg-zinc-50 dark:bg-zinc-900 p-3 rounded-lg overflow-x-auto text-sm shadow-sm border border-zinc-200 dark:border-zinc-800 
                          [&>code]:!bg-transparent [&>code]:p-0 
                          [&>code]:text-zinc-800 dark:[&>code]:text-zinc-200 
                          [&>code_.hljs-keyword]:text-purple-600 dark:[&>code_.hljs-keyword]:text-purple-400 
                          [&>code_.hljs-string]:text-emerald-600 dark:[&>code_.hljs-string]:text-emerald-400 
                          [&>code_.hljs-number]:text-amber-600 dark:[&>code_.hljs-number]:text-amber-400 
                          [&>code_.hljs-comment]:text-zinc-500 dark:[&>code_.hljs-comment]:text-zinc-500 
                          [&>code_.hljs-operator]:text-sky-600 dark:[&>code_.hljs-operator]:text-sky-400 
                          [&>code_.hljs-function]:text-blue-600 dark:[&>code_.hljs-function]:text-blue-300
                          [&>code_.hljs-title]:text-blue-600 dark:[&>code_.hljs-title]:text-blue-300
                          [&>code_.hljs-params]:text-zinc-800 dark:[&>code_.hljs-params]:text-zinc-300
                          [&>code_.hljs-variable]:text-orange-600 dark:[&>code_.hljs-variable]:text-orange-300
                          [&>code_.hljs-class]:text-yellow-600 dark:[&>code_.hljs-class]:text-yellow-300"
                          {...props}
                        />
                      ),
                      code: ({
                        inline,
                        className,
                        children,
                        ...props
                      }: CodeProps) =>
                        inline ? (
                          <code
                            className="bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded text-sm text-zinc-800 dark:text-zinc-200"
                            {...props}
                          >
                            {children}
                          </code>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        ),
                      h1: ({ ...props }) => (
                        <h1
                          className="text-3xl font-extrabold mb-8 mt-2 text-zinc-800 dark:text-zinc-100 tracking-tight border-b pb-4 border-zinc-200 dark:border-zinc-800"
                          {...props}
                        />
                      ),
                      h2: ({ ...props }) => (
                        <h2
                          className="text-2xl font-bold mb-6 mt-10 text-zinc-800 dark:text-zinc-100 tracking-tight relative before:content-['#'] before:absolute before:-left-5 before:text-indigo-400 dark:before:text-indigo-500 before:opacity-0 hover:before:opacity-100 before:transition-opacity"
                          {...props}
                        />
                      ),
                      h3: ({ ...props }) => (
                        <h3
                          className="text-xl font-semibold mb-4 mt-8 text-zinc-700 dark:text-zinc-200 tracking-tight"
                          {...props}
                        />
                      ),
                      h4: ({ ...props }) => (
                        <h4
                          className="text-lg font-medium mb-3 mt-6 text-zinc-700 dark:text-zinc-300 tracking-tight"
                          {...props}
                        />
                      ),
                      p: ({ ...props }) => (
                        <p
                          className="mb-3 leading-relaxed text-sm text-gray-700 dark:text-gray-300"
                          {...props}
                        />
                      ),
                      ul: ({ ...props }) => (
                        <ul
                          className="list-disc pl-6 mb-3 space-y-1.5 text-gray-700 dark:text-gray-300"
                          {...props}
                        />
                      ),
                      ol: ({ ...props }) => (
                        <ol
                          className="list-decimal pl-6 mb-3 space-y-1.5 text-gray-700 dark:text-gray-300"
                          {...props}
                        />
                      ),
                      li: ({ children, ...props }) => (
                        <li className="mb-1 pl-1 text-sm" {...props}>
                          <div className="inline">{children}</div>
                        </li>
                      ),
                      a: ({ ...props }) => (
                        <a
                          className="text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 text-sm no-underline hover:underline"
                          {...props}
                        />
                      ),
                      blockquote: ({ ...props }) => (
                        <blockquote
                          className="border-l-4 border-gray-200 dark:border-zinc-700 pl-3 italic my-3 text-sm text-gray-600 dark:text-gray-400"
                          {...props}
                        />
                      ),
                    }}
                  >
                    {output}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
            <div className="absolute bottom-0 right-0 left-0 pointer-events-none p-4">
              <ScrollToBottomButton
                show={showScrollButton}
                onClick={() => scrollToBottom()}
              />
            </div>
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
