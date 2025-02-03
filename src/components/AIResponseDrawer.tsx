"use client";

import { Drawer } from "vaul";
import { useAIResponse } from "@/lib/hooks/use-ai-response";
import { Loader2, Sparkles } from "lucide-react";
import { Suspense } from "react";
import { MagicalLoadingOverlay } from "./MagicalLoadingOverlay";
import { useScrollControl } from "@/lib/hooks/use-scroll-control";
import { ScrollToBottomButton } from "./ScrollToBottomButton";
import dynamic from "next/dynamic";
const MarkdownRenderer = dynamic(
  () => import("@/components/MarkdownRenderer"),
  {
    ssr: false,
  }
);

interface AIResponseDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  option: string;
  selectedText: string;
  markdownContent: string;
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
        <Drawer.Overlay className="fixed inset-0 bg-black/40 backdrop-blur-[2px] transition-opacity duration-300" />
        <Drawer.Content
          aria-describedby={undefined}
          className="fixed bottom-0 left-0 right-0 mt-24 flex h-[80dvh] flex-col rounded-t-[10px] bg-white dark:bg-zinc-900 focus:outline-none focus-visible:outline-none transform transition-transform duration-300 ease-out will-change-transform"
          style={{
            transform: "translate3d(0, 0, 0)",
            backfaceVisibility: "hidden",
            WebkitBackfaceVisibility: "hidden",
          }}
        >
          <Drawer.Title className="sr-only">AI Response</Drawer.Title>
          <div className="absolute inset-0 z-50 pointer-events-none rounded-t-[10px] overflow-hidden">
            <MagicalLoadingOverlay isLoading={isLoading} />
          </div>
          <div className="relative flex-1">
            <div
              ref={containerRef}
              className="absolute inset-0 overflow-y-auto rounded-t-[10px] bg-zinc-100 dark:bg-zinc-800 p-4 scroll-smooth overscroll-contain"
              style={{
                overscrollBehavior: "contain",
                WebkitOverflowScrolling: "touch",
              }}
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
                  <Suspense
                    fallback={
                      <div className="animate-pulse space-y-4">
                        <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-3/4"></div>
                        <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded"></div>
                        <div className="h-4 bg-zinc-200 dark:bg-zinc-700 rounded w-5/6"></div>
                      </div>
                    }
                  >
                    <MarkdownRenderer content={output} />
                  </Suspense>
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
