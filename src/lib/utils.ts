import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface DebounceOptions {
  leading?: boolean;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  options: DebounceOptions = {}
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let lastArgs: Parameters<T> | undefined;

  return function debounced(this: any, ...args: Parameters<T>) {
    const context = this;

    // If leading is true and there's no timeout, execute immediately
    if (options.leading && !timeoutId) {
      func.apply(context, args);
    }

    lastArgs = args;

    // Clear the existing timeout (if any)
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    // Set up new timeout
    timeoutId = setTimeout(() => {
      if (!options.leading || timeoutId) {
        func.apply(context, lastArgs!);
      }
      timeoutId = undefined;
    }, wait);
  };
}
