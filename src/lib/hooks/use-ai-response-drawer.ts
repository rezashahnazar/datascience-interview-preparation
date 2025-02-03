import { useCallback, useEffect } from "react";

type UseAIResponseDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function useAIResponseDrawer({
  isOpen,
  onClose,
}: UseAIResponseDrawerProps) {
  // Add touch event listeners for smoother interactions on mobile
  useEffect(() => {
    if (isOpen) {
      // Prevent body scroll when drawer is open
      document.body.style.overflow = "hidden";

      // Add touch-action manipulation for smoother touch handling
      document.documentElement.style.touchAction = "none";

      return () => {
        document.body.style.overflow = "";
        document.documentElement.style.touchAction = "";
      };
    }
  }, [isOpen]);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) {
        // Add a small delay before closing to allow animations to complete
        requestAnimationFrame(() => {
          onClose();
        });
      }
    },
    [onClose]
  );

  return {
    isOpen,
    handleOpenChange,
  };
}
