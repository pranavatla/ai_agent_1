"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { FaGithub, FaLinkedinIn } from "react-icons/fa6";
import { site, socials } from "@/lib/site";
import { useMedia, useReducedMotion } from "@/lib/media";

export default function Hero() {
  const reduced = useReducedMotion();
  // The full name is too long for one readable line on a phone, so it breaks before the surname.
  const wide = useMedia("(min-width: 640px)");
  const lines = wide ? [site.name.toUpperCase()] : ["SAI PRANAV", "ATLA"];
  const svgRef = useRef<SVGSVGElement>(null);
  const textRef = useRef<SVGTextElement>(null);
  const [box, setBox] = useState<string | null>(null);

  useEffect(() => {
    let raf = 0;
    let cancelled = false;
    document.fonts.ready.then(() => {
      const t = textRef.current;
      const svg = svgRef.current;
      if (!t || !svg || cancelled) return;
      const b = t.getBBox();
      setBox(`${b.x - 6} ${b.y - 6} ${b.width + 12} ${b.height + 12}`);
      // The viewBox scales the text to fit, so size the stroke in user units to land at ~1.4px on screen.
      t.style.strokeWidth = `${(1.4 * (b.width + 12)) / svg.clientWidth}`;
      // A glyph outline is longer than its advance width, so double the measured length
      // or the dash pattern would leave gaps in the finished letters.
      const len = t.getComputedTextLength() * 2;
      t.style.strokeDasharray = `${len}`;
      if (reduced) {
        t.style.strokeDashoffset = "0";
        t.style.fillOpacity = "1";
        return;
      }
      const start = performance.now();
      const tick = (now: number) => {
        const k = Math.min(1, (now - start) / 3000);
        t.style.strokeDashoffset = `${len * (1 - (1 - Math.pow(1 - k, 3)))}`;
        if (k < 1) raf = requestAnimationFrame(tick); // draws once, never loops
        else t.style.fillOpacity = "1"; // then the letters fill in for legibility
      };
      raf = requestAnimationFrame(tick);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [reduced, wide]);

  return (
    <section id="home" className="relative flex min-h-svh flex-col items-center justify-center px-6 text-center">
      <h1 className="w-full max-w-7xl">
        <span className="sr-only">{site.name}</span>
        <svg
          ref={svgRef}
          aria-hidden
          viewBox={box ?? "0 0 1000 220"}
          className="mx-auto h-auto w-[min(90vw,80rem)] overflow-visible"
          style={{ opacity: box ? 1 : 0 }}
        >
          <defs>
            <linearGradient id="hero-stroke" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#1d6fd0" />
              <stop offset="100%" stopColor="#6d3bd4" />
            </linearGradient>
          </defs>
          <text
            ref={textRef}
            x="500"
            y="180"
            textAnchor="middle"
            fill="url(#hero-stroke)"
            stroke="url(#hero-stroke)"
            style={{ fontFamily: "var(--font-display)", fontSize: 200, fontWeight: 800, letterSpacing: "-0.01em", fillOpacity: 0, transition: "fill-opacity 1s ease" }}
          >
            {lines.map((line, i) => (
              <tspan key={line} x="500" dy={i ? 220 : 0}>
                {line}
              </tspan>
            ))}
          </text>
        </svg>
      </h1>
      <motion.p
        className="mt-10 font-mono text-xs tracking-[0.22em] text-deep uppercase"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.8 }}
      >
        {site.handle}
      </motion.p>
      <motion.p
        className="mt-4 max-w-2xl text-base text-slate-600 sm:text-lg lg:max-w-none"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.8 }}
      >
        {site.tagline}
      </motion.p>
      <motion.div
        className="mt-9 flex flex-wrap items-center justify-center gap-x-6 gap-y-3"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.1, duration: 0.8 }}
      >
        <a
          href={site.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 rounded-full bg-ink px-6 py-3 text-sm font-medium text-white shadow-[0_10px_24px_-10px_rgba(15,23,42,0.6)] transition duration-200 ease-snappy hover:bg-deep active:scale-[0.97]"
        >
          <FaLinkedinIn aria-hidden className="size-4" />
          LinkedIn
        </a>
        <a
          href={socials.find((s) => s.label === "GitHub")?.href}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-medium text-slate-800 shadow-sm transition duration-200 ease-snappy hover:border-deep/40 hover:text-deep active:scale-[0.97]"
        >
          <FaGithub aria-hidden className="size-4 text-deep" />
          GitHub
        </a>
      </motion.div>
    </section>
  );
}
