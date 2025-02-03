import { useState, useCallback, useRef, useEffect } from "react";

type DataStreamState = {
  output: string;
  error: string | null;
  isLoading: boolean;
};

type StreamParams = {
  option: string;
  selectedText: string;
  markdownContent: string;
};

export function useDataStream() {
  const [state, setState] = useState<DataStreamState>({
    output: "",
    error: null,
    isLoading: false,
  });
  const activeStreamRef = useRef<boolean>(false);
  const controllerRef = useRef<AbortController | null>(null);

  const processStreamLine = (line: string) => {
    if (!line) return;

    const [type, ...rest] = line.split(":");
    const content = rest.join(":"); // Rejoin in case the content contains colons

    switch (type) {
      case "0": // Text content
        if (content) {
          try {
            const text = JSON.parse(content);
            setState((prev) => ({
              ...prev,
              output: prev.output + text,
            }));
          } catch (e) {
            console.warn("Failed to parse text content:", content);
            console.error(e);
          }
        }
        break;

      case "1": // Error content
        if (content) {
          try {
            const errorData = JSON.parse(content);
            setState((prev) => ({
              ...prev,
              error: errorData.message || "Stream error",
            }));
          } catch (e) {
            console.warn("Failed to parse error content:", content);
            console.error(e);
          }
        }
        break;

      case "2": // Data content (ignore for now)
        break;

      case "f": // First chunk
        break;

      case "e": // End event
        break;

      case "d": // Done - this is the final message
        setState((prev) => ({
          ...prev,
          isLoading: false,
        }));
        break;

      default:
        break;
    }
  };

  const startStream = useCallback(
    async ({ option, selectedText, markdownContent }: StreamParams) => {
      // If there's an existing controller, abort it
      if (controllerRef.current) {
        controllerRef.current.abort();
      }

      // Prevent multiple streams from running simultaneously
      if (activeStreamRef.current) {
        return;
      }

      activeStreamRef.current = true;

      // Create new controller
      controllerRef.current = new AbortController();
      const signal = controllerRef.current.signal;

      setState({
        output: "",
        error: null,
        isLoading: true,
      });

      try {
        const response = await fetch("/api/ai-stream", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({ option, selectedText, markdownContent }),
          signal,
        });

        if (!response.body) throw new Error("No response body");

        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .getReader();

        while (true) {
          const { done, value } = await reader.read();

          // Process any remaining data even if done is true
          if (value) {
            value.split("\n").forEach(processStreamLine);
          }

          // Only break after processing the final chunk
          if (done) {
            setState((prev) => ({
              ...prev,
              isLoading: false,
            }));
            break;
          }
        }
      } catch (error) {
        setState((prev) => ({
          ...prev,
          error: error instanceof Error ? error.message : "Stream failed",
          isLoading: false,
        }));
      } finally {
        activeStreamRef.current = false;
      }
    },
    []
  );

  // Cleanup function
  useEffect(() => {
    return () => {
      if (controllerRef.current) {
        controllerRef.current.abort();
      }
    };
  }, []);

  return { ...state, startStream };
}
