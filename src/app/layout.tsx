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
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body className={`${inter.className} h-full`}>
        <CustomThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex h-full bg-background text-foreground">
            {/* Navigation sidebar */}
            <Navigation items={navigationItems} />
            {/* Main content */}
            <main className="w-full overflow-y-auto px-5 pb-4 pt-16 lg:p-6">
              {children}
            </main>

            {/* AI Response sidebar container */}
            <div
              className="flex-none w-0 transition-[width] duration-300 ease-spring bg-white dark:bg-zinc-900 shadow-xl border-l border-zinc-200 dark:border-zinc-800"
              data-sidebar-container
            />
          </div>
          {/* Portal container for dropdowns */}
          <div
            id="dropdown-portal-container"
            className="fixed inset-0 pointer-events-none z-[99999]"
          />
        </CustomThemeProvider>
      </body>
    </html>
  );
}
