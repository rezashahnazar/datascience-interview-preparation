import { useCallback, useEffect, useRef, useState } from "react";

const SCROLL_THRESHOLD = 100; // pixels from bottom

export function useScrollControl(isLoading: boolean) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const wasNearBottomRef = useRef(true);

  const isNearBottom = useCallback(() => {
    const container = containerRef.current;
    if (!container) return true;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    return distanceFromBottom <= SCROLL_THRESHOLD;
  }, []);

  const scrollToBottom = useCallback((smooth = true) => {
    const container = containerRef.current;
    if (!container) return;

    const scrollHeight = container.scrollHeight;
    container.scrollTo({
      top: scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }, []);

  const handleScroll = useCallback(() => {
    const isNearBottomNow = isNearBottom();
    setShowScrollButton(!isNearBottomNow);
    wasNearBottomRef.current = isNearBottomNow;
  }, [isNearBottom]);

  // Handle content changes
  const handleContentChange = useCallback(() => {
    requestAnimationFrame(() => {
      const container = containerRef.current;
      if (!container) return;

      // Update scroll button visibility
      handleScroll();

      // Auto-scroll if we were near bottom
      if (isLoading && wasNearBottomRef.current) {
        scrollToBottom(false);
      }
    });
  }, [handleScroll, isLoading, scrollToBottom]);

  // Watch for content changes
  useEffect(() => {
    const container = containerRef.current;
    const content = contentRef.current;
    if (!container || !content) return;

    // Initial scroll check
    handleScroll();

    // Set up content observer
    const observer = new MutationObserver(handleContentChange);
    observer.observe(content, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    // Set up scroll listener
    container.addEventListener("scroll", handleScroll);

    return () => {
      container.removeEventListener("scroll", handleScroll);
      observer.disconnect();
    };
  }, [handleScroll, handleContentChange]);

  // Watch for loading state changes
  useEffect(() => {
    if (isLoading && wasNearBottomRef.current) {
      handleContentChange();
    }
  }, [isLoading, handleContentChange]);

  return {
    containerRef,
    contentRef,
    showScrollButton,
    scrollToBottom,
  };
}
