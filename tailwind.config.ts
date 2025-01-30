import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: "var(--primary)",
        secondary: "var(--secondary)",
        accent: "var(--accent)",
        error: "var(--error)",
        success: "var(--success)",
        warning: "var(--warning)",
        info: "var(--info)",
        muted: "var(--muted)",
        "light-border": "var(--light-border)",
        "light-bg-alt": "var(--light-bg-alt)",
      },
      boxShadow: {
        custom: "0 4px 6px var(--shadow)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
  darkMode: "class",
} satisfies Config;
