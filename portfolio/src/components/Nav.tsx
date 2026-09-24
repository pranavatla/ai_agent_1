"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Briefcase, EnvelopeSimple, House, List, Moon, SquaresFour, Sun, User, X } from "@phosphor-icons/react/dist/ssr";
import { sections } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";
import { setTheme, useDark } from "@/lib/theme";

const icons = { home: House, work: Briefcase, services: SquaresFour, about: User, contact: EnvelopeSimple };
const MIDLINE = { rootMargin: "-50% 0px -50% 0px" };
const pill =
  "pointer-events-none absolute right-full mr-3 translate-x-2 rounded-full bg-surface px-3 py-1.5 text-xs font-medium whitespace-nowrap text-slate-800 opacity-0 shadow-[0_6px_18px_rgba(15,23,42,0.14)] transition duration-200 ease-snappy group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:translate-x-0 group-focus-visible:opacity-100";

// Light/dark switch. Shows where it will take you: a moon on the light theme, a sun on the dark one.
function ThemeToggle({ className, withLabel }: { className: string; withLabel?: boolean }) {
  const dark = useDark();
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  const Icon = dark ? Sun : Moon;
  return (
    <button type="button" aria-label={label} onClick={() => setTheme(dark ? "light" : "dark")} className={`group ${className}`}>
      <Icon aria-hidden className="size-[18px]" />
      {withLabel && (
        <span aria-hidden className={pill}>
          {dark ? "Light theme" : "Dark theme"}
        </span>
      )}
    </button>
  );
}

export default function Nav() {
  const reduced = useReducedMotion();
  const [active, setActive] = useState("home");
  const [open, setOpen] = useState(false);

  // Active link. The footer maps to Contact, or the rail goes stale at the very bottom.
  useEffect(() => {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) setActive(e.target.id === "footer" ? "contact" : e.target.id);
    }, MIDLINE);
    [...sections.map((s) => s.id), "footer"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    // Keep the page still behind the full-screen menu.
    document.documentElement.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.documentElement.style.overflow = "";
    };
  }, [open]);

  const btn = "bg-surface text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.12)] hover:text-deep";
  const current = "bg-deep text-white shadow-[0_6px_18px_rgba(29,111,208,0.4)]";

  return (
    <>
      {/* Desktop rail: section links, then a short divider and the theme switch. */}
      <div className="fixed top-1/2 right-5 z-50 hidden -translate-y-1/2 flex-col items-center gap-3 md:flex">
      <nav aria-label="Sections" className="flex flex-col gap-3">
        {sections.map(({ id, label }) => {
          const Icon = icons[id];
          const isCurrent = active === id;
          return (
            <a
              key={id}
              href={`#${id}`}
              aria-label={label}
              aria-current={isCurrent ? "location" : undefined}
              className={`group relative grid size-11 place-items-center rounded-full transition duration-200 ease-snappy active:scale-[0.94] ${isCurrent ? current : btn}`}
            >
              <Icon aria-hidden className="size-[18px]" />
              <span aria-hidden className={pill}>
                {label}
              </span>
            </a>
          );
        })}
      </nav>
        <span aria-hidden className="h-px w-5 bg-slate-300" />
        <ThemeToggle withLabel className={`relative grid size-11 place-items-center rounded-full transition duration-200 ease-snappy active:scale-[0.94] ${btn}`} />
      </div>

      {/* Phones: the theme switch sits beside the menu button. */}
      <ThemeToggle className={`fixed top-4 right-[4.25rem] z-[60] grid size-11 place-items-center rounded-full transition duration-200 ease-snappy active:scale-[0.94] md:hidden ${btn}`} />

      <button
        type="button"
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        aria-controls="mobile-menu"
        onClick={() => setOpen((o) => !o)}
        className={`fixed top-4 right-4 z-[60] grid size-11 place-items-center rounded-full transition duration-200 ease-snappy active:scale-[0.94] md:hidden ${btn}`}
      >
        {open ? <X aria-hidden className="size-5" /> : <List aria-hidden className="size-5" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.nav
            id="mobile-menu"
            aria-label="Sections"
            className="fixed inset-0 z-[55] flex flex-col justify-center gap-2 bg-ground/95 px-8 backdrop-blur-xl md:hidden"
            style={{ transformOrigin: "calc(100% - 2.25rem) 2.25rem" }} // anchored to the hamburger button that opened it
            initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.96 }}
            animate={reduced ? { opacity: 1 } : { opacity: 1, scale: 1 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.98 }}
            transition={reduced ? { duration: 0.15 } : { type: "spring", bounce: 0.1, duration: 0.35 }}
          >
            {sections.map(({ id, label }, i) => (
              <motion.a
                key={id}
                href={`#${id}`}
                onClick={() => setOpen(false)}
                aria-current={active === id ? "location" : undefined}
                initial={reduced ? { opacity: 0 } : { opacity: 0, y: 18 }}
                animate={reduced ? { opacity: 1 } : { opacity: 1, y: 0 }}
                transition={reduced ? { duration: 0.15 } : { delay: 0.05 * i + 0.08, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="font-display py-2 text-4xl font-extrabold tracking-tight text-slate-700 aria-[current]:text-deep"
              >
                {label}
              </motion.a>
            ))}
          </motion.nav>
        )}
      </AnimatePresence>
    </>
  );
}
