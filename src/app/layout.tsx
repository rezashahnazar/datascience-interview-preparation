import type { Metadata } from "next";
import "./globals.css";
import Navigation from "@/components/Navigation";
import { getAllDocs } from "@/utils/markdown";

export const metadata: Metadata = {
  title: "Data Science Learning",
  description: "A comprehensive guide to data science interview preparation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const docs = getAllDocs();
  const navigationItems = docs.map((doc) => ({
    slug: doc.id,
    title: doc.title,
  }));

  return (
    <html lang="en">
      <body className="min-h-screen bg-white dark:bg-zinc-900">
        <div className="flex min-h-screen">
          <Navigation items={navigationItems} />
          <main className="flex-1 px-5 py-4 pt-16 lg:pt-4 lg:p-6 lg:ml-56 w-full overflow-x-hidden bg-white dark:bg-zinc-900">
            <div className="max-w-3xl mx-auto">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
