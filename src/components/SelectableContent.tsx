"use client";

import { ReactNode } from "react";
import SelectionDropdown from "./SelectionDropdown";

interface SelectableContentProps {
  children: ReactNode;
  originalContent: string;
}

export default function SelectableContent({
  children,
  originalContent,
}: SelectableContentProps) {
  const findMarkdownContent = (selectedText: string): string => {
    // Split the selected text and original content into lines
    const selectedLines = selectedText.split("\n");
    const contentLines = originalContent.split("\n");

    // Find the first and last lines in the original content
    let startLineIndex = -1;
    let endLineIndex = -1;

    const cleanMarkdown = (text: string) => {
      return text
        .replace(/^[#\s-]+/, "") // Remove heading markers and list markers
        .replace(/\*\*/g, "") // Remove bold markers
        .replace(/\*/g, "") // Remove italic markers
        .replace(/`/g, "") // Remove code markers
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Remove link formatting
        .replace(/^>/g, "") // Remove blockquote markers
        .trim();
    };

    for (let i = 0; i < contentLines.length; i++) {
      const line = contentLines[i];
      const cleanLine = cleanMarkdown(line);
      const cleanSelectedLine = cleanMarkdown(selectedLines[0]);

      // Find the first line that matches
      if (startLineIndex === -1 && cleanLine === cleanSelectedLine) {
        startLineIndex = i;
      }

      // Find the last line that matches
      const lastSelectedLine = cleanMarkdown(
        selectedLines[selectedLines.length - 1]
      );
      if (cleanMarkdown(line).includes(lastSelectedLine)) {
        endLineIndex = i;
        // Only break if we've found both the start and a matching end after it
        if (startLineIndex !== -1 && endLineIndex >= startLineIndex) {
          // If this is a partial line selection, check if there's more content
          const cleanLastSelected = cleanMarkdown(
            selectedLines[selectedLines.length - 1]
          );
          const cleanCurrentLine = cleanMarkdown(line);
          if (cleanCurrentLine.length > cleanLastSelected.length) {
            // Only break if this is the best match so far
            if (cleanCurrentLine.indexOf(cleanLastSelected) === 0) {
              break;
            }
          } else {
            break;
          }
        }
      }
    }

    // If we found valid start and end positions, extract the markdown
    if (
      startLineIndex !== -1 &&
      endLineIndex !== -1 &&
      endLineIndex >= startLineIndex
    ) {
      // Return the original markdown content with formatting
      return contentLines.slice(startLineIndex, endLineIndex + 1).join("\n");
    }

    // Fallback to the selected text if we couldn't find a match
    return selectedText;
  };

  const handleOptionSelect = (option: string) => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const selectedText = selection.toString().trim();
    const markdownContent = findMarkdownContent(selectedText);

    console.log(`Selected option: ${option}`);
    console.log(`Selected text: ${selectedText}`);
    console.log(`Selected markdown: ${markdownContent}`);
  };

  return (
    <div className="relative min-h-full" data-markdown-content>
      <SelectionDropdown onOptionSelect={handleOptionSelect} />
      <div className="prose dark:prose-invert max-w-none">{children}</div>
    </div>
  );
}
