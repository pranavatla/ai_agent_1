"use client";

import { motion } from "motion/react";
import { EnvelopeSimple, FileArrowDown, LinkedinLogo } from "@phosphor-icons/react";
import { site } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

export default function Hero() {
  const reduced = useReducedMotion();

  return (
    <section
      id="home"
      className="relative flex min-h-[calc(100dvh-4rem)] pt-16 flex-col items-center justify-center px-6 text-center"
    >
      <div className="mx-auto max-w-4xl">
        <motion.p
          initial={reduced ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="font-mono text-xs font-medium tracking-[0.2em] text-deep uppercase"
        >
          {site.handle}
        </motion.p>

        <motion.h1
          initial={reduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="font-display mt-5 text-5xl font-extrabold tracking-tight text-slate-900 sm:text-7xl lg:text-8xl"
        >
          {site.name}
        </motion.h1>

        <motion.p
          initial={reduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-slate-600 sm:text-lg"
        >
          {site.tagline}
        </motion.p>

        <motion.div
          initial={reduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-8 flex flex-wrap items-center justify-center gap-3"
        >
          <a
            href="#resume"
            className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-medium text-white shadow-sm transition duration-150 hover:bg-deep active:scale-[0.97]"
          >
            <FileArrowDown size={18} />
            <span>View Résumé</span>
          </a>
          <a
            href={site.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-medium text-slate-800 shadow-xs transition duration-150 hover:border-deep/40 hover:text-deep active:scale-[0.97]"
          >
            <LinkedinLogo size={18} weight="fill" className="text-deep" />
            <span>LinkedIn</span>
          </a>
          <a
            href={`mailto:${site.email}`}
            className="inline-flex items-center gap-2 rounded-full border border-transparent px-4 py-3 text-sm font-medium text-slate-600 transition duration-150 hover:text-deep"
          >
            <EnvelopeSimple size={18} />
            <span>{site.email}</span>
          </a>
        </motion.div>
      </div>
    </section>
  );
}
