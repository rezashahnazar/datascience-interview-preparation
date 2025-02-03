import { getDocBySlug } from "@/utils/markdown";
import MarkdownContent from "@/components/MarkdownContent";

export default function Home() {
  const doc = getDocBySlug("main");

  return (
    <article className="py-6 max-w-3xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">{doc.title}</h1>
      <MarkdownContent content={doc.content} />
    </article>
  );
}
