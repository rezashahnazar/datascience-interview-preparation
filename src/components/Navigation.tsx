"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import ThemeToggle from "./ThemeToggle";

interface NavigationItem {
  slug: string;
  title: string;
}

interface NavigationProps {
  items: NavigationItem[];
}

export default function Navigation({ items }: NavigationProps) {
  const pathname = usePathname();
  const navigationItems = items.filter((item) => item.slug !== "main");
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile Navbar */}
      <div className="fixed top-0 left-0 right-0 z-40 bg-white dark:bg-zinc-900 border-b border-gray-100 dark:border-zinc-800 lg:hidden backdrop-blur-sm bg-white/90 dark:bg-zinc-900/90">
        <div className="flex items-center h-12 px-3">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
            aria-label="Toggle menu"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {isOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
          <Link
            href="/"
            className={`ml-3 text-base font-bold truncate ${
              pathname === "/"
                ? "text-indigo-600 dark:text-indigo-400"
                : "text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400"
            }`}
          >
            Data Science Learning
          </Link>
        </div>
      </div>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 backdrop-blur-[2px] z-40 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <nav
        className={`fixed top-0 left-0 h-screen w-[280px] lg:w-56 bg-white dark:bg-zinc-900 p-3 overflow-y-auto transform transition-transform duration-300 ease-in-out z-50 border-r border-gray-100 dark:border-zinc-800 shadow-lg flex flex-col ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0 lg:top-0`}
      >
        <div className="mb-6 hidden lg:block">
          <Link
            href="/"
            className={`block text-xl font-bold ${
              pathname === "/"
                ? "text-indigo-600 dark:text-indigo-400"
                : "text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400"
            }`}
            onClick={() => setIsOpen(false)}
          >
            Data Science Learning
          </Link>
        </div>
        <ul className="space-y-1.5 mt-6 lg:mt-0 flex-1">
          {navigationItems.map((item) => {
            const isActive = pathname === `/${item.slug}`;
            return (
              <li key={item.slug}>
                <Link
                  href={`/${item.slug}`}
                  onClick={() => setIsOpen(false)}
                  className={`block px-3 py-1.5 rounded-lg transition-colors text-sm ${
                    isActive
                      ? "bg-indigo-600 text-white dark:bg-indigo-500"
                      : "text-zinc-600 dark:text-zinc-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:text-indigo-600 dark:hover:text-indigo-400"
                  }`}
                >
                  {item.title}
                </Link>
              </li>
            );
          })}
        </ul>
        <div className="mt-6 pt-6 border-t border-gray-100 dark:border-zinc-800">
          <ThemeToggle />
        </div>
      </nav>
    </>
  );
}
