import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface ScrollToBottomButtonProps {
  onClick: () => void;
  show: boolean;
}

export function ScrollToBottomButton({
  onClick,
  show,
}: ScrollToBottomButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "pointer-events-auto float-right flex h-10 w-10 items-center justify-center rounded-full bg-indigo-500 text-white shadow-lg transition-all duration-200 hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2 dark:bg-indigo-600 dark:hover:bg-indigo-700 dark:focus:ring-indigo-500",
        show
          ? "translate-y-0 opacity-100"
          : "translate-y-4 opacity-0 pointer-events-none"
      )}
      aria-label="Scroll to bottom"
    >
      <ChevronDown className="h-5 w-5" />
    </button>
  );
}
