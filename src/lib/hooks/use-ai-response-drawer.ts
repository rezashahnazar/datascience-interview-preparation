import { useCallback } from "react";

type UseAIResponseDrawerProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function useAIResponseDrawer({
  isOpen,
  onClose,
}: UseAIResponseDrawerProps) {
  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (!open) onClose();
    },
    [onClose]
  );

  return {
    isOpen,
    handleOpenChange,
  };
}
