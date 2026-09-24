"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Briefcase, House, LayoutGrid, Mail, Menu, User, X } from "lucide-react";
import { sections } from "@/lib/site";

const icons = { home: House, work: Briefcase, services: LayoutGrid, about: User, contact: Mail };
const MIDLINE = { rootMargin: "-50% 0px -50% 0px" };

export default function Nav() {
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
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const btn = "bg-white text-slate-700 shadow-[0_6px_18px_rgba(15,23,42,0.12)] hover:text-deep";
  const current = "bg-deep text-white shadow-[0_6px_18px_rgba(29,111,208,0.4)]";

  return (
    <>
      <nav aria-label="Sections" className="fixed top-1/2 right-5 z-50 hidden -translate-y-1/2 flex-col gap-3 md:flex">
        {sections.map(({ id, label }) => {
          const Icon = icons[id];
          const isCurrent = active === id;
          return (
            <a
              key={id}
              href={`#${id}`}
              aria-label={label}
              aria-current={isCurrent ? "location" : undefined}
              className={`group relative grid size-11 place-items-center rounded-full transition-colors duration-300 ${isCurrent ? current : btn}`}
            >
              <Icon aria-hidden className="size-[18px]" />
              <span
                aria-hidden
                className="pointer-events-none absolute right-full mr-3 translate-x-2 rounded-full bg-white px-3 py-1.5 font-mono text-[11px] tracking-[0.14em] whitespace-nowrap text-slate-800 uppercase opacity-0 shadow-[0_6px_18px_rgba(15,23,42,0.14)] transition duration-300 group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:translate-x-0 group-focus-visible:opacity-100"
              >
                {label}
              </span>
            </a>
          );
        })}
      </nav>

      <button
        type="button"
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        aria-controls="mobile-menu"
        onClick={() => setOpen((o) => !o)}
        className={`fixed top-4 right-4 z-[60] grid size-11 place-items-center rounded-full transition-colors md:hidden ${btn}`}
      >
        {open ? <X aria-hidden className="size-5" /> : <Menu aria-hidden className="size-5" />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.nav
            id="mobile-menu"
            aria-label="Sections"
            className="fixed inset-0 z-[55] flex flex-col justify-center gap-2 bg-ground/95 px-8 backdrop-blur-xl md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {sections.map(({ id, label }, i) => (
              <motion.a
                key={id}
                href={`#${id}`}
                onClick={() => setOpen(false)}
                aria-current={active === id ? "location" : undefined}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.06 * i + 0.05, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="font-display py-2 text-4xl font-bold tracking-[-0.02em] text-slate-700 uppercase aria-[current]:text-deep"
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
