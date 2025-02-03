"use client";

import { ReactNode, useState } from "react";
import SelectionDropdown from "./SelectionDropdown";
import AIResponseDrawer from "./AIResponseDrawer";
import AIResponseSidebar from "./AIResponseSidebar";
import { useMediaQuery } from "@/lib/hooks/use-media-query";

interface SelectableContentProps {
  children: ReactNode;
  originalContent: string;
}

export default function SelectableContent({
  children,
  originalContent,
}: SelectableContentProps) {
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const [drawerState, setDrawerState] = useState<{
    isOpen: boolean;
    option: string;
    selectedText: string;
    markdownContent: string;
    selectionRect: DOMRect | null;
  }>({
    isOpen: false,
    option: "",
    selectedText: "",
    markdownContent: "",
    selectionRect: null,
  });

  const findMarkdownContent = (selectedText: string): string => {
    if (!originalContent || !selectedText) return "";

    const trimmedSelection = selectedText.trim();
    if (!trimmedSelection) return "";

    // Split into lines for better matching
    const selectedLines = trimmedSelection
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (selectedLines.length === 0) return "";

    // Find the first occurrence of the first selected line
    const firstLine = selectedLines[0];
    const lastLine = selectedLines[selectedLines.length - 1];

    const contentLines = originalContent.split("\n");
    let startLineIndex = -1;
    let endLineIndex = -1;

    // Helper function to clean line for comparison
    const cleanLine = (line: string) =>
      line
        .trim()
        .replace(/^[`\s]+|[`\s]+$/g, "") // Remove backticks and whitespace
        .replace(/^\s*[-*+]\s+/, "") // Remove list markers
        .replace(/^\s*\d+\.\s+/, ""); // Remove numbered list markers

    // Find the start line with cleaned comparison
    const cleanFirstLine = cleanLine(firstLine);
    for (let i = 0; i < contentLines.length; i++) {
      if (cleanLine(contentLines[i]).includes(cleanFirstLine)) {
        startLineIndex = i;
        break;
      }
    }

    // If we couldn't find the start, return the original selection
    if (startLineIndex === -1) return selectedText;

    // Find the end line with cleaned comparison
    const cleanLastLine = cleanLine(lastLine);
    for (let i = startLineIndex; i < contentLines.length; i++) {
      if (cleanLine(contentLines[i]).includes(cleanLastLine)) {
        endLineIndex = i;
        break;
      }
    }

    // If we couldn't find the end, use the start line
    if (endLineIndex === -1) endLineIndex = startLineIndex;

    // Check if we're inside a code block
    let isInCodeBlock = false;
    for (let i = 0; i < startLineIndex; i++) {
      if (contentLines[i].trim().startsWith("```")) {
        isInCodeBlock = !isInCodeBlock;
      }
    }

    // If in code block, expand selection to include the entire block
    if (isInCodeBlock) {
      while (
        startLineIndex > 0 &&
        !contentLines[startLineIndex - 1].trim().startsWith("```")
      ) {
        startLineIndex--;
      }
      if (startLineIndex > 0) startLineIndex--; // Include the opening ```

      while (
        endLineIndex < contentLines.length - 1 &&
        !contentLines[endLineIndex + 1].trim().startsWith("```")
      ) {
        endLineIndex++;
      }
      if (endLineIndex < contentLines.length - 1) endLineIndex++; // Include the closing ```
    } else {
      // Regular text: Include context within the same section
      while (
        startLineIndex > 0 &&
        contentLines[startLineIndex - 1].trim() !== ""
      ) {
        if (contentLines[startLineIndex - 1].startsWith("#")) break;
        startLineIndex--;
      }

      while (
        endLineIndex < contentLines.length - 1 &&
        contentLines[endLineIndex + 1].trim() !== ""
      ) {
        if (contentLines[endLineIndex + 1].startsWith("#")) break;
        endLineIndex++;
      }
    }

    return contentLines
      .slice(startLineIndex, endLineIndex + 1)
      .join("\n")
      .trim();
  };

  const handleOptionSelect = (option: string) => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    // Get the selected text before any state changes
    const selectedText = selection.toString().trim();
    const markdownContent = findMarkdownContent(selectedText);
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    // Open drawer/sidebar with the selected content
    setDrawerState({
      isOpen: true,
      option,
      selectedText,
      markdownContent,
      selectionRect: rect,
    });

    // Clear the selection
    selection.removeAllRanges();
  };

  return (
    <div className="relative min-h-full" data-markdown-content>
      <SelectionDropdown onOptionSelect={handleOptionSelect} />
      <div className="prose dark:prose-invert max-w-none">{children}</div>
      {isDesktop ? (
        <AIResponseSidebar
          isOpen={drawerState.isOpen}
          onClose={() => setDrawerState((prev) => ({ ...prev, isOpen: false }))}
          option={drawerState.option}
          selectedText={drawerState.selectedText}
          markdownContent={drawerState.markdownContent}
          selectionRect={drawerState.selectionRect}
        />
      ) : (
        <AIResponseDrawer
          isOpen={drawerState.isOpen}
          onClose={() => setDrawerState((prev) => ({ ...prev, isOpen: false }))}
          option={drawerState.option}
          selectedText={drawerState.selectedText}
          markdownContent={drawerState.markdownContent}
        />
      )}
    </div>
  );
}
