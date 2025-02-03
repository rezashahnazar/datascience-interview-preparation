"use client";

import { useEffect, useState, useRef, useCallback } from "react";

interface Position {
  top: number;
  left: number;
}

interface SelectionDropdownProps {
  onOptionSelect: (option: string) => void;
}

export default function SelectionDropdown({
  onOptionSelect,
}: SelectionDropdownProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<Position>({ top: 0, left: 0 });
  const [isSelecting, setIsSelecting] = useState(false);
  const [lastSelectionText, setLastSelectionText] = useState<string>("");
  const [isToggling, setIsToggling] = useState(false);
  const currentSelectionRef = useRef<Selection | null>(null);

  const options = [
    { id: "rewrite", title: "Rewrite" },
    { id: "describe", title: "Describe More" },
    { id: "example", title: "Give Example" },
  ];

  const calculatePosition = useCallback((rect: DOMRect): Position => {
    const MARGIN = 10;
    const DROPDOWN_HEIGHT = 120; // Approximate height of dropdown with all options
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // Always position the dropdown above the selection on iOS
    const isIOS =
      /iPad|iPhone|iPod/.test(navigator.userAgent) &&
      !/CriOS|FxiOS/.test(navigator.userAgent) &&
      "ontouchend" in document;

    // Calculate vertical position (always above on iOS)
    let top = isIOS
      ? Math.max(MARGIN, rect.top - DROPDOWN_HEIGHT - MARGIN * 2)
      : rect.bottom + MARGIN;

    // If not iOS and not enough space below, position above
    if (!isIOS && top + DROPDOWN_HEIGHT > viewportHeight) {
      top = Math.max(MARGIN, rect.top - DROPDOWN_HEIGHT - MARGIN);
    }

    // Calculate horizontal position
    const left = Math.max(
      MARGIN,
      Math.min(
        rect.left + rect.width / 2 - 80, // 80 is half of min-w-[160px]
        viewportWidth - 160 - MARGIN // Ensure dropdown doesn't go off-screen horizontally
      )
    );

    return { top, left };
  }, []);

  const checkSelection = useCallback(() => {
    // Don't check selection if we're toggling the dropdown
    if (isToggling) return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) {
      // Keep dropdown visible if clicking on existing selection
      const text = selection?.toString().trim() || "";
      if (text === lastSelectionText && text) {
        return;
      }
      setIsVisible(false);
      return;
    }

    const text = selection.toString().trim();
    if (!text) {
      setIsVisible(false);
      return;
    }

    setLastSelectionText(text);
    currentSelectionRef.current = selection;
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    setPosition(calculatePosition(rect));
    setIsVisible(true);
  }, [calculatePosition, lastSelectionText, isToggling]);

  const handleSelectionStart = useCallback(() => {
    if (!("ontouchstart" in window)) {
      setIsVisible(false);
      currentSelectionRef.current = null;
    }
  }, []);

  const handleTouchStart = useCallback(() => {
    const selection = window.getSelection();
    const text = selection?.toString().trim() || "";

    // If touching on existing selection, toggle dropdown
    if (text && text === lastSelectionText) {
      setIsToggling(true);
      setIsVisible((prev) => !prev);

      // Reset toggling state after a short delay
      setTimeout(() => {
        setIsToggling(false);
      }, 300);
      return;
    }

    // Otherwise, start new selection
    setIsSelecting(true);
    setIsVisible(false);
  }, [lastSelectionText]);

  const handleTouchEnd = useCallback(() => {
    if (isToggling) return; // Don't process touch end during toggle

    setIsSelecting(false);
    // Check selection immediately after touch end
    setTimeout(() => {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) {
        const text = selection.toString().trim();
        if (text) {
          setLastSelectionText(text);
          currentSelectionRef.current = selection;
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          setPosition(calculatePosition(rect));
          setIsVisible(true);
        }
      }
    }, 100);
  }, [calculatePosition, isToggling]);

  useEffect(() => {
    // For desktop: only show dropdown after mouseup (selection complete)
    document.addEventListener("mouseup", checkSelection);
    document.addEventListener("mousedown", handleSelectionStart);

    // For touch devices
    document.addEventListener("touchend", handleTouchEnd);
    document.addEventListener("touchstart", handleTouchStart);

    // Handle selection changes for both desktop and touch
    const handleSelectionChange = () => {
      // Don't process selection changes during toggle or when dropdown is intentionally hidden
      if (!isToggling && "ontouchstart" in window && !isSelecting) {
        const selection = window.getSelection();
        const text = selection?.toString().trim() || "";
        // Only update if we have a new selection that's different from the current one
        if (text && text !== lastSelectionText) {
          checkSelection();
        }
      }
    };
    document.addEventListener("selectionchange", handleSelectionChange);

    return () => {
      document.removeEventListener("mouseup", checkSelection);
      document.removeEventListener("mousedown", handleSelectionStart);
      document.removeEventListener("touchend", handleTouchEnd);
      document.removeEventListener("touchstart", handleTouchStart);
      document.removeEventListener("selectionchange", handleSelectionChange);
    };
  }, [
    checkSelection,
    handleTouchEnd,
    handleSelectionStart,
    handleTouchStart,
    isSelecting,
    isToggling,
    lastSelectionText,
  ]);

  if (!isVisible) return null;

  return (
    <div
      className="fixed z-[9999] bg-white dark:bg-zinc-800 rounded-lg shadow-lg border border-zinc-200 dark:border-zinc-700 py-1 min-w-[160px] select-none animate-fade-in [&_*]:!select-none [-webkit-touch-callout:none] [-webkit-user-select:none] [user-select:none]"
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
        WebkitTouchCallout: "none",
        WebkitUserSelect: "none",
        userSelect: "none",
      }}
    >
      {options.map((option) => (
        <button
          key={option.id}
          className="w-full px-4 py-2 text-left text-sm hover:bg-zinc-100 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 select-none"
          onMouseDown={(e) => {
            e.preventDefault(); // Prevent selection from being cleared
            onOptionSelect(option.id);
            setIsVisible(false);
          }}
          onTouchStart={(e) => {
            e.preventDefault(); // Prevent native touch selection
            onOptionSelect(option.id);
            setIsVisible(false);
          }}
        >
          {option.title}
        </button>
      ))}
    </div>
  );
}
