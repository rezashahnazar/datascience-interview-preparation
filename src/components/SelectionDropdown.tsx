"use client";

import { useEffect, useState, useRef } from "react";

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
  const currentSelectionRef = useRef<Selection | null>(null);

  const options = [
    { id: "rewrite", title: "Rewrite" },
    { id: "describe", title: "Describe More" },
    { id: "example", title: "Give Example" },
  ];

  const calculatePosition = (rect: DOMRect): Position => {
    const MARGIN = 10;
    const DROPDOWN_HEIGHT = 120; // Approximate height of dropdown with all options
    const viewportHeight = window.innerHeight;

    // Calculate initial positions
    let top = rect.bottom + MARGIN;
    const left = Math.max(
      MARGIN,
      Math.min(
        rect.left + rect.width / 2 - 80, // 80 is half of min-w-[160px]
        window.innerWidth - 160 - MARGIN // Ensure dropdown doesn't go off-screen horizontally
      )
    );

    // Check if dropdown would go below viewport
    if (top + DROPDOWN_HEIGHT > viewportHeight) {
      // Position dropdown above the selection instead
      top = Math.max(MARGIN, rect.top - DROPDOWN_HEIGHT - MARGIN);
    }

    return { top, left };
  };

  const checkSelection = () => {
    setTimeout(() => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) {
        setIsVisible(false);
        return;
      }

      const text = selection.toString().trim();
      if (!text) {
        setIsVisible(false);
        return;
      }

      currentSelectionRef.current = selection;
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      setPosition(calculatePosition(rect));
      setIsVisible(true);
    }, 0);
  };

  const handleSelectionStart = () => {
    setIsVisible(false);
    currentSelectionRef.current = null;
  };

  const handleTouchStart = (e: TouchEvent) => {
    // Check if the touch target is a text node or element with text
    const target = e.target as HTMLElement;
    if (target.nodeType === Node.TEXT_NODE || target.innerText?.trim()) {
      setIsSelecting(true);
    }
  };

  const handleTouchEnd = (e: TouchEvent) => {
    if (!isSelecting) return;

    setIsSelecting(false);
    checkSelection();
  };

  useEffect(() => {
    document.addEventListener("mouseup", checkSelection);
    document.addEventListener("mousedown", handleSelectionStart);
    document.addEventListener("touchend", handleTouchEnd);
    document.addEventListener("touchstart", handleTouchStart);

    const handleSelectionChange = () => {
      if ("ontouchstart" in window && isSelecting) {
        checkSelection();
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
  }, [isSelecting]);

  if (!isVisible) return null;

  return (
    <div
      className="fixed z-50 bg-white dark:bg-zinc-800 rounded-lg shadow-lg border border-zinc-200 dark:border-zinc-700 py-1 min-w-[160px] select-none animate-fade-in"
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
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
        >
          {option.title}
        </button>
      ))}
    </div>
  );
}
