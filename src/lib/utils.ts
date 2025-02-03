import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface DebounceOptions {
  leading?: boolean;
}

export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  func: T,
  wait: number,
  options: DebounceOptions = {}
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let lastArgs: Parameters<T> | undefined;

  return function debounced(...args: Parameters<T>) {
    // Using arrow function to preserve this context
    const executeFunction = () => {
      func(...args);
      timeoutId = undefined;
    };

    // If leading is true and there's no timeout, execute immediately
    if (options.leading && !timeoutId) {
      executeFunction();
      return;
    }

    // Clear the existing timeout
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    // Set up new timeout
    timeoutId = setTimeout(executeFunction, wait);
  };
}
