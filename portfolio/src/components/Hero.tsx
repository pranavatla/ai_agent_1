"use client";

import { motion } from "motion/react";
import { GithubLogo, LinkedinLogo } from "@phosphor-icons/react/dist/ssr";
import { site, socials } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

const button =
  "inline-flex items-center gap-2 rounded-full border border-slate-200 bg-surface px-6 py-3 text-sm font-medium text-slate-800 shadow-sm transition duration-200 ease-snappy hover:border-deep/40 hover:text-deep active:scale-[0.97]";

export default function Hero() {
  const reduced = useReducedMotion();
  const rise = (delay: number) => ({
    initial: reduced ? false : { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay },
  });

  // Left-aligned in the page's content column; the right side stays open over the background band.
  return (
    <section id="home" className="relative flex min-h-[100dvh] items-center px-6 sm:px-10 md:pr-24">
      <div className="mx-auto w-full max-w-6xl">
        <motion.ul {...rise(0)} className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs font-medium tracking-[0.2em] text-deep uppercase">
          {site.handle.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </motion.ul>
        <motion.h1 {...rise(0.1)} className="font-display mt-5 text-5xl font-extrabold tracking-tight text-slate-900 sm:text-7xl lg:text-8xl">
          {site.name}
        </motion.h1>
        <motion.p {...rise(0.2)} className="mt-6 max-w-2xl text-base leading-relaxed text-slate-600 sm:text-lg">
          {site.tagline}
        </motion.p>
        <motion.div {...rise(0.3)} className="mt-8 flex flex-wrap items-center gap-3">
          <a href={site.linkedin} target="_blank" rel="noopener noreferrer" className={button}>
            <LinkedinLogo aria-hidden weight="fill" className="size-4 text-deep" />
            LinkedIn
          </a>
          <a href={socials.find((s) => s.label === "GitHub")?.href} target="_blank" rel="noopener noreferrer" className={button}>
            <GithubLogo aria-hidden weight="fill" className="size-4 text-deep" />
            GitHub
          </a>
        </motion.div>
      </div>
    </section>
  );
}
