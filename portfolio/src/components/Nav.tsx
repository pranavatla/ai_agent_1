"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { FileArrowDown, List, X } from "@phosphor-icons/react";
import { sections, site } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

const MIDLINE = { rootMargin: "-30% 0px -60% 0px" };

export default function Nav() {
  const reduced = useReducedMotion();
  const [active, setActive] = useState<string>("work");
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) setActive(e.target.id);
      }
    }, MIDLINE);

    sections.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) io.observe(el);
    });

    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    document.documentElement.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.documentElement.style.overflow = "";
    };
  }, [open]);

  return (
    <>
      <header
        className={`fixed top-0 inset-x-0 z-50 h-16 transition-all duration-200 ${
          scrolled
            ? "border-b border-slate-200/80 bg-white/85 shadow-xs backdrop-blur-md"
            : "border-b border-transparent bg-transparent"
        }`}
      >
        <div className="mx-auto flex h-full max-w-6xl items-center justify-between px-6 sm:px-10">
          <a
            href="#home"
            className="font-display text-lg font-bold tracking-tight text-slate-900 transition hover:text-deep"
          >
            {site.brand}
            <span className="text-deep font-mono text-sm ml-1">.in</span>
          </a>

          {/* Desktop single-line navigation */}
          <nav aria-label="Main navigation" className="hidden items-center gap-1 md:flex">
            {sections.map(({ id, label }) => {
              const isCurrent = active === id;
              return (
                <a
                  key={id}
                  href={`#${id}`}
                  className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition duration-150 ${
                    isCurrent
                      ? "text-deep font-semibold"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/60"
                  }`}
                >
                  {label}
                </a>
              );
            })}
          </nav>

          <div className="hidden items-center gap-3 md:flex">
            <a
              href="#resume"
              className="inline-flex items-center gap-1.5 rounded-full bg-slate-900 px-4 py-2 text-xs font-medium text-white shadow-xs transition duration-150 hover:bg-deep active:scale-[0.97]"
            >
              <FileArrowDown size={15} />
              <span>Résumé</span>
            </a>
          </div>

          {/* Mobile hamburger button */}
          <button
            type="button"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((o) => !o)}
            className="grid size-10 place-items-center rounded-lg border border-slate-200 bg-white text-slate-700 shadow-xs transition hover:text-deep active:scale-95 md:hidden"
          >
            {open ? <X size={20} /> : <List size={20} />}
          </button>
        </div>
      </header>

      {/* Mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            id="mobile-nav"
            aria-label="Mobile navigation"
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: -8 }}
            animate={reduced ? { opacity: 1 } : { opacity: 1, y: 0 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 flex flex-col justify-between bg-white/95 px-6 pt-24 pb-8 backdrop-blur-xl md:hidden"
          >
            <nav className="flex flex-col gap-2">
              {sections.map(({ id, label }) => (
                <a
                  key={id}
                  href={`#${id}`}
                  onClick={() => setOpen(false)}
                  className={`py-3 text-2xl font-bold tracking-tight transition ${
                    active === id ? "text-deep" : "text-slate-800 hover:text-deep"
                  }`}
                >
                  {label}
                </a>
              ))}
            </nav>

            <div className="border-t border-slate-200 pt-6">
              <a
                href="#resume"
                onClick={() => setOpen(false)}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 py-3.5 text-sm font-semibold text-white shadow-xs transition hover:bg-deep active:scale-[0.98]"
              >
                <FileArrowDown size={18} />
                <span>View Résumés</span>
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
