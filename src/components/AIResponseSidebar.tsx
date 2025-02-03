"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useAIResponse } from "@/lib/hooks/use-ai-response";
import { Sparkles, X } from "lucide-react";
import { Suspense, useEffect } from "react";
import { MagicalLoadingOverlay } from "./MagicalLoadingOverlay";
import { useScrollControl } from "@/lib/hooks/use-scroll-control";
import { ScrollToBottomButton } from "./ScrollToBottomButton";
import { createPortal } from "react-dom";
import dynamic from "next/dynamic";

const MarkdownRenderer = dynamic(
  () => import("@/components/MarkdownRenderer"),
  {
    ssr: false,
  }
);

interface AIResponseSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  option: string;
  selectedText: string;
  markdownContent: string;
  selectionRect: DOMRect | null;
}

export default function AIResponseSidebar({
  isOpen,
  onClose,
  option,
  selectedText,
  markdownContent,
  selectionRect,
}: AIResponseSidebarProps) {
  const { output, error, isLoading, handleOpenChange } = useAIResponse({
    isOpen,
    onClose,
    option,
    selectedText,
    markdownContent,
  });

  const { containerRef, contentRef, showScrollButton, scrollToBottom } =
    useScrollControl(isLoading);

  // Handle sidebar container width
  useEffect(() => {
    const sidebarContainer = document.querySelector("[data-sidebar-container]");

    if (sidebarContainer) {
      if (isOpen) {
        sidebarContainer.classList.add("!w-[400px]");
      } else {
        sidebarContainer.classList.remove("!w-[400px]");
      }
    }

    return () => {
      sidebarContainer?.classList.remove("!w-[400px]");
    };
  }, [isOpen]);

  // Handle mobile body scroll lock
  useEffect(() => {
    if (!isOpen) return;

    const isMobile = window.matchMedia("(max-width: 1024px)").matches;
    if (isMobile) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [isOpen]);

  const sidebarContainer =
    typeof document !== "undefined"
      ? document.querySelector("[data-sidebar-container]")
      : null;

  if (!sidebarContainer) return null;

  return createPortal(
    <AnimatePresence mode="wait">
      {isOpen && (
        <>
          {/* Overlay for mobile only */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/20 backdrop-blur-[2px] z-[9998] lg:hidden"
            onClick={() => handleOpenChange(false)}
          />

          {/* Sidebar */}
          <motion.div
            initial={{
              opacity: 0,
              ...(selectionRect
                ? {
                    height: selectionRect.height,
                    width: 0,
                    x: "100%",
                    y: selectionRect.top + window.scrollY - 80,
                    originX: 1,
                  }
                : {
                    height: "100%",
                    width: 0,
                    x: "100%",
                    originX: 1,
                  }),
            }}
            animate={{
              opacity: 1,
              height: "100%",
              width: "400px",
              x: 0,
              y: 0,
              originX: 1,
            }}
            exit={{
              opacity: 0,
              width: 0,
              x: "100%",
              ...(selectionRect
                ? {
                    height: selectionRect.height,
                    y: selectionRect.top + window.scrollY - 80,
                  }
                : {
                    height: "100%",
                  }),
              originX: 1,
            }}
            transition={{
              type: "spring",
              damping: 25,
              stiffness: 250,
            }}
            style={{
              position: "fixed",
              top: 0,
              right: 0,
              bottom: 0,
              width: "400px",
              zIndex: 2147483646, // One less than the maximum possible z-index
            }}
            className="bg-white dark:bg-zinc-900 shadow-xl overflow-hidden origin-right flex flex-col"
          >
            {/* Close Button */}
            <button
              onClick={() => handleOpenChange(false)}
              className="absolute top-2 right-3 z-[10000] p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
              aria-label="Close sidebar"
            >
              <X className="w-5 h-5 text-zinc-500 dark:text-zinc-400" />
            </button>

            <div className="absolute inset-0 z-[9999] pointer-events-none overflow-hidden">
              <MagicalLoadingOverlay isLoading={isLoading} />
            </div>

            {/* Header */}
            <div className="flex-none px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-indigo-500" />
                <h2 className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-xl font-bold text-transparent">
                  Magical Assistant
                </h2>
              </div>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto min-h-0 bg-zinc-100 dark:bg-zinc-800 overscroll-contain">
              <div ref={containerRef} className="p-4">
                <div className="mx-auto max-w-2xl pb-4">
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
            </div>

            {/* Footer */}
            <div className="flex-none h-[50px] px-4 border-t border-zinc-200 dark:border-zinc-800 flex items-center justify-between">
              <AnimatePresence mode="wait">
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="flex items-center gap-2 text-sm"
                  >
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 blur-lg opacity-50" />
                      <div className="relative px-3 py-1 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 text-zinc-100">
                        Crafting magic...
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              <ScrollToBottomButton
                show={showScrollButton}
                onClick={() => scrollToBottom()}
              />
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    sidebarContainer
  );
}
