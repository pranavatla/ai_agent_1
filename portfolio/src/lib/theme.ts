import { useSyncExternalStore } from "react";

export const isDark = () => document.documentElement.dataset.theme === "dark";

export function subscribeTheme(cb: () => void) {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => mo.disconnect();
}

export const useDark = () => useSyncExternalStore(subscribeTheme, isDark, () => false);

export function setTheme(theme: "light" | "dark") {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("theme", theme);
  } catch {}
  // Keep the browser toolbar colour in step with the manual choice.
  document.querySelectorAll('meta[name="theme-color"]').forEach((m) => m.setAttribute("content", theme === "dark" ? "#0b0f1a" : "#f7f8fc"));
}
