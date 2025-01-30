"use client";

import dynamic from "next/dynamic";

const ThemeProvider = dynamic(
  () => import("next-themes").then((mod) => mod.ThemeProvider),
  {
    ssr: false,
  }
);

export default function CustomThemeProvider({
  children,
  ...props
}: {
  children: React.ReactNode;
  [key: string]: unknown;
}) {
  return <ThemeProvider {...props}>{children}</ThemeProvider>;
}
