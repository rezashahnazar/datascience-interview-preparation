"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Pencil, Info, BookText } from "lucide-react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

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
    { id: "rewrite", title: "Rewrite", icon: Pencil },
    { id: "describe", title: "Describe More", icon: Info },
    { id: "example", title: "Give Example", icon: BookText },
  ];

  const calculatePosition = useCallback((rect: DOMRect): Position => {
    const MARGIN = 10;
    const DROPDOWN_HEIGHT = 120; // Approximate height of dropdown with all options
    const IOS_ACTION_BAR_HEIGHT = 50; // Approximate height of iOS action bar
    const viewportHeight = window.innerHeight;
    const viewportWidth = window.innerWidth;

    // Detect iOS Safari
    const isIOS =
      /iPad|iPhone|iPod/.test(navigator.userAgent) &&
      !/CriOS|FxiOS/.test(navigator.userAgent) &&
      "ontouchend" in document;

    // Calculate space available above and below selection
    const spaceAbove = rect.top;
    const spaceBelow = viewportHeight - rect.bottom;
    const effectiveSpaceBelow = isIOS
      ? spaceBelow - IOS_ACTION_BAR_HEIGHT
      : spaceBelow;

    // Determine if we should position above or below
    const shouldPositionAbove =
      effectiveSpaceBelow < DROPDOWN_HEIGHT + MARGIN &&
      spaceAbove > DROPDOWN_HEIGHT + MARGIN;

    // Calculate vertical position
    const top = shouldPositionAbove
      ? Math.max(MARGIN, rect.top - DROPDOWN_HEIGHT - MARGIN)
      : rect.bottom + MARGIN;

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

    // Get the selection's container element
    const range = selection.getRangeAt(0);
    const container = range.commonAncestorContainer.parentElement;

    // Check if the selection is within the sidebar
    const sidebarContainer = document.querySelector("[data-sidebar-container]");
    const isWithinSidebar = sidebarContainer?.contains(container);

    // Get the correct rect based on whether it's in the sidebar or not
    const rect = range.getBoundingClientRect();

    // Adjust position if within sidebar to ensure dropdown appears above it
    const adjustedPosition = calculatePosition(rect);
    if (isWithinSidebar) {
      // Force the dropdown to appear on the left side of the sidebar
      adjustedPosition.left = Math.min(
        adjustedPosition.left,
        rect.left - 170 // 160px (dropdown width) + 10px margin
      );
    }

    setLastSelectionText(text);
    currentSelectionRef.current = selection;
    setPosition(adjustedPosition);
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

  const portalContainer = document.getElementById("dropdown-portal-container");
  if (!portalContainer) return null;

  return createPortal(
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.15 }}
          className="fixed bg-white dark:bg-zinc-900 rounded-lg shadow-lg border border-zinc-200 dark:border-zinc-800 p-1.5 min-w-[160px] pointer-events-auto isolate"
          style={{
            position: "fixed",
            top: `${position.top}px`,
            left: `${position.left}px`,
            zIndex: 999999,
            transform: "translate3d(0, 0, 0)",
            willChange: "transform",
          }}
        >
          {options.map((option) => (
            <button
              key={option.id}
              className="w-full px-4 py-2 text-left text-sm hover:bg-zinc-100 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 select-none flex items-center gap-2"
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
              <option.icon className="w-4 h-4" />
              {option.title}
            </button>
          ))}
        </motion.div>
      )}
    </AnimatePresence>,
    portalContainer
  );
}
