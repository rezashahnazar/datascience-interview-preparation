import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  content: string;
}

interface CodeProps extends React.ClassAttributes<HTMLElement> {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export default function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:text-zinc-800 dark:prose-headings:text-zinc-200 prose-a:text-indigo-600 dark:prose-a:text-indigo-400 prose-a:no-underline hover:prose-a:text-indigo-700 hover:dark:prose-a:text-indigo-300 hover:prose-a:underline">
      <ReactMarkdown
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ ...props }) => (
            <pre
              className="bg-gray-50 dark:bg-zinc-900 p-3 rounded-lg overflow-x-auto text-sm shadow-sm border border-gray-100 dark:border-zinc-800"
              {...props}
            />
          ),
          code: ({ inline, ...props }: CodeProps) =>
            inline ? (
              <code
                className="bg-gray-50 dark:bg-zinc-900 px-1 py-0.5 rounded text-sm"
                {...props}
              />
            ) : (
              <code {...props} />
            ),
          h1: ({ ...props }) => (
            <h1
              className="text-3xl font-extrabold mb-8 mt-2 text-zinc-800 dark:text-zinc-100 tracking-tight border-b pb-4 border-zinc-200 dark:border-zinc-800"
              {...props}
            />
          ),
          h2: ({ ...props }) => (
            <h2
              className="text-2xl font-bold mb-6 mt-10 text-zinc-800 dark:text-zinc-100 tracking-tight relative before:content-['#'] before:absolute before:-left-5 before:text-indigo-400 dark:before:text-indigo-500 before:opacity-0 hover:before:opacity-100 before:transition-opacity"
              {...props}
            />
          ),
          h3: ({ ...props }) => (
            <h3
              className="text-xl font-semibold mb-4 mt-8 text-zinc-700 dark:text-zinc-200 tracking-tight"
              {...props}
            />
          ),
          h4: ({ ...props }) => (
            <h4
              className="text-lg font-medium mb-3 mt-6 text-zinc-700 dark:text-zinc-300 tracking-tight"
              {...props}
            />
          ),
          p: ({ ...props }) => (
            <p
              className="mb-3 leading-relaxed text-sm text-gray-700 dark:text-gray-300"
              {...props}
            />
          ),
          ul: ({ ...props }) => (
            <ul
              className="list-disc pl-6 mb-3 space-y-1.5 text-gray-700 dark:text-gray-300"
              {...props}
            />
          ),
          ol: ({ ...props }) => (
            <ol
              className="list-decimal pl-6 mb-3 space-y-1.5 text-gray-700 dark:text-gray-300"
              {...props}
            />
          ),
          li: ({ children, ...props }) => (
            <li className="mb-1 pl-1 text-sm" {...props}>
              <div className="inline">{children}</div>
            </li>
          ),
          a: ({ ...props }) => (
            <a
              className="text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 text-sm no-underline hover:underline"
              {...props}
            />
          ),
          blockquote: ({ ...props }) => (
            <blockquote
              className="border-l-4 border-gray-200 dark:border-zinc-700 pl-3 italic my-3 text-sm text-gray-600 dark:text-gray-400"
              {...props}
            />
          ),
          table: ({ ...props }) => (
            <div className="overflow-x-auto my-4 rounded-lg border border-gray-100 dark:border-zinc-800 shadow-sm">
              <table
                className="min-w-full divide-y divide-gray-100 dark:divide-zinc-800 text-sm"
                {...props}
              />
            </div>
          ),
          th: ({ ...props }) => (
            <th
              className="px-3 py-2 bg-gray-50 dark:bg-zinc-900 text-sm font-semibold text-gray-700 dark:text-gray-300"
              {...props}
            />
          ),
          td: ({ ...props }) => (
            <td
              className="px-3 py-2 border-t border-gray-100 dark:border-zinc-800 text-sm text-gray-600 dark:text-gray-400"
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
