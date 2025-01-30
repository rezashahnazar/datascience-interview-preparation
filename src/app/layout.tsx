import type { Metadata } from "next";
import "./globals.css";
import Navigation from "@/components/Navigation";
import { getAllDocs } from "@/utils/markdown";
import CustomThemeProvider from "@/components/ThemeProvider";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Data Science Learning",
  description: "A comprehensive guide to data science interview preparation",
  icons: {
    icon: [
      {
        url: "/favicon.ico",
        sizes: "any",
      },
      {
        url: "/favicon.svg",
        type: "image/svg+xml",
      },
    ],
  },
  authors: [
    {
      name: "Reza Shahnazar",
      url: "https://github.com/rezashahnazar",
    },
  ],
  creator: "Reza Shahnazar",
  publisher: "Digikala",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
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
      <body className={inter.className}>
        <CustomThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-dvh bg-background text-foreground">
            <Navigation items={navigationItems} />
            <main className="flex-1 px-5 py-4 pt-16 lg:pt-4 lg:p-6 lg:ml-56 w-full overflow-x-hidden">
              <div className="max-w-3xl mx-auto">{children}</div>
            </main>
          </div>
        </CustomThemeProvider>
      </body>
    </html>
  );
}
