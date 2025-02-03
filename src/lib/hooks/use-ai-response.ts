import { useEffect } from "react";
import { useAIResponseDrawer } from "./use-ai-response-drawer";
import { useDataStream } from "./use-data-stream";

type UseAIResponseProps = {
  isOpen: boolean;
  onClose: () => void;
  option: string;
  selectedText: string;
  markdownContent: string;
};

export function useAIResponse({
  isOpen,
  onClose,
  option,
  selectedText,
  markdownContent,
}: UseAIResponseProps) {
  // Pure drawer behavior
  const { handleOpenChange } = useAIResponseDrawer({
    isOpen,
    onClose,
  });

  // Pure streaming behavior
  const { output, error, isLoading, startStream } = useDataStream();

  // Coordination logic
  useEffect(() => {
    if (isOpen && option && selectedText && markdownContent) {
      startStream({ option, selectedText, markdownContent });
    }
  }, [isOpen, option, selectedText, markdownContent, startStream]);

  return {
    output,
    error,
    isLoading,
    handleOpenChange,
  };
}
