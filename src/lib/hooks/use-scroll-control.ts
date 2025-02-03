import { useCallback, useEffect, useRef, useState } from "react";
import { debounce } from "@/lib/utils";

export function useScrollControl(isLoading: boolean) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const scrollTimeoutRef = useRef<number | undefined>(undefined);

  const checkScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShowScrollButton(!isNearBottom);
  }, []);

  // Optimized scroll handler with debounce
  const debouncedCheckScroll = useCallback(
    debounce(checkScroll, 100, { leading: true }),
    [checkScroll]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Use passive event listener for better performance
    container.addEventListener("scroll", debouncedCheckScroll, {
      passive: true,
    });

    return () => {
      container.removeEventListener("scroll", debouncedCheckScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [debouncedCheckScroll]);

  // Smooth scroll to bottom with requestAnimationFrame
  const scrollToBottom = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const targetScroll = container.scrollHeight - container.clientHeight;
    const startScroll = container.scrollTop;
    const distance = targetScroll - startScroll;
    const duration = 300; // ms
    const startTime = performance.now();

    const animateScroll = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function for smooth animation
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);

      container.scrollTop = startScroll + distance * easeOutCubic;

      if (progress < 1) {
        requestAnimationFrame(animateScroll);
      }
    };

    requestAnimationFrame(animateScroll);
  }, []);

  // Auto-scroll to bottom when loading
  useEffect(() => {
    if (isLoading) {
      const container = containerRef.current;
      if (!container) return;

      // Clear any existing scroll timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      // Schedule scroll with RAF for smoother performance
      scrollTimeoutRef.current = window.setTimeout(() => {
        requestAnimationFrame(() => {
          container.scrollTop = container.scrollHeight;
        });
      }, 100);
    }
  }, [isLoading]);

  return {
    containerRef,
    contentRef,
    showScrollButton,
    scrollToBottom,
  };
}
