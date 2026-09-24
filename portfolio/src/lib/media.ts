import { useSyncExternalStore } from "react";

// Hydration-safe media query: the server snapshot is used for SSR, the real value right after.
export function useMedia(query: string) {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(query);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export const useReducedMotion = () => useMedia("(prefers-reduced-motion: reduce)");
export const useIsLg = () => useMedia("(min-width: 1024px)");
