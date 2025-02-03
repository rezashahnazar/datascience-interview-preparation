import { getDocBySlug, getAllDocSlugs } from "@/utils/markdown";
import MarkdownContent from "@/components/MarkdownContent";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  const paths = getAllDocSlugs();
  return paths;
}

export default async function Page({
  params,
}: {
  params: Promise<{
    slug: string;
  }>;
}) {
  try {
    const doc = await getDocBySlug((await params).slug);

    return (
      <article className="py-6 max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">{doc.title}</h1>
        <MarkdownContent content={doc.content} />
      </article>
    );
  } catch (error) {
    console.error(error);
    notFound();
  }
}
