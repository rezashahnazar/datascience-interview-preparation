"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { SunIcon, MoonIcon } from "@heroicons/react/24/solid";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Ensure the component is mounted before rendering to avoid hydration mismatches
  useEffect(() => setMounted(true), []);

  if (!mounted) return null;

  return (
    <button
      aria-label="Toggle Dark Mode"
      type="button"
      className="w-10 h-6 flex items-center bg-gray-300 dark:bg-gray-600 rounded-full p-1 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      <span className="sr-only">Toggle Dark Mode</span>
      <SunIcon
        className={`w-4 h-4 text-yellow-500 transition-transform duration-300 ${
          theme === "dark" ? "transform translate-x-4" : ""
        }`}
      />
      <MoonIcon
        className={`w-4 h-4 text-indigo-500 transition-transform duration-300 ${
          theme === "dark" ? "" : "transform -translate-x-4"
        }`}
      />
    </button>
  );
}
