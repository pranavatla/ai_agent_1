"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { motion, useMotionValue, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { LinkedinLogo } from "@phosphor-icons/react";
import { about, experience, recognition, site, socials, timeline } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

const words = about.bio.split(" ");
// Overlapping slices (2.2x each word's own step) so the highlight reads as a wave, not a ticking counter.
const STEP = 1 / (words.length + 1.2);

function BioWord({ word, i, progress }: { word: string; i: number; progress: MotionValue<number> }) {
  const from = i * STEP;
  const to = from + 2.2 * STEP;
  const color = useTransform(progress, [from, to], ["rgba(51,65,85,0.35)", "rgba(51,65,85,1)"]);
  return <motion.span style={{ color }}>{word} </motion.span>;
}

export default function About() {
  const reduced = useReducedMotion();
  const bioRef = useRef<HTMLParagraphElement>(null);
  const { scrollYProgress: bioProgress } = useScroll({ target: bioRef, offset: ["start 0.9", "end 0.5"] });

  // Pointer-parallax tilt on the portrait, spring-damped so it settles instead of snapping.
  const px = useMotionValue(0);
  const py = useMotionValue(0);
  const rotateX = useSpring(useTransform(py, [-0.5, 0.5], [6, -6]), { stiffness: 150, damping: 18 });
  const rotateY = useSpring(useTransform(px, [-0.5, 0.5], [-6, 6]), { stiffness: 150, damping: 18 });
  const portraitRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (reduced) return;
    const el = portraitRef.current;
    if (!el) return;
    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      px.set((e.clientX - r.left) / r.width - 0.5);
      py.set((e.clientY - r.top) / r.height - 0.5);
    };
    const onLeave = () => {
      px.set(0);
      py.set(0);
    };
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced, px, py]);

  return (
    <section id="about" className="scroll-mt-20 px-6 py-24 sm:px-10">
      <div className="mx-auto max-w-6xl">
        {/* Profile & Narrative */}
        <div className="grid gap-12 lg:grid-cols-12 lg:gap-16 items-center">
          {/* Portrait Column */}
          <div className="lg:col-span-5 flex justify-center">
            <motion.div
              ref={portraitRef}
              initial={reduced ? false : { opacity: 0, y: 24, scale: 0.96 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              style={reduced ? undefined : { rotateX, rotateY, transformPerspective: 800 }}
              className="relative aspect-[896/1195] w-full max-w-[340px] sm:max-w-[380px] overflow-hidden rounded-2xl border border-slate-200/90 bg-white p-2 shadow-sm"
            >
              <div className="relative h-full w-full overflow-hidden rounded-xl bg-slate-100">
                <Image
                  src={site.portrait}
                  alt={`Portrait of ${site.name}`}
                  fill
                  sizes="(min-width: 1024px) 380px, 70vw"
                  className="object-cover"
                  priority
                />
              </div>
            </motion.div>
          </div>

          {/* Statement & Bio Column */}
          <motion.div
            initial={reduced ? false : { opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-7 flex flex-col justify-center"
          >
            <span className="font-mono text-xs font-medium tracking-[0.2em] text-deep uppercase">
              About
            </span>

            <h2 className="font-display mt-4 text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              Systems that hold.
            </h2>

            <p ref={bioRef} className="mt-6 text-lg leading-relaxed">
              {reduced ? about.bio : words.map((w, i) => <BioWord key={i} word={w} i={i} progress={bioProgress} />)}
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <a
                href={site.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-5 py-2.5 text-xs font-semibold text-white shadow-xs transition hover:bg-deep active:scale-[0.97]"
              >
                <LinkedinLogo size={16} weight="fill" />
                <span>Connect on LinkedIn</span>
              </a>

              <div className="flex items-center gap-2">
                {socials.map(({ label, href, icon: Icon }) => (
                  <a
                    key={label}
                    href={href}
                    target={href.startsWith("http") ? "_blank" : undefined}
                    rel="noopener noreferrer"
                    aria-label={label}
                    className="grid size-9 place-items-center rounded-full border border-slate-200 bg-white text-slate-700 shadow-xs transition hover:border-deep/50 hover:text-deep active:scale-95"
                  >
                    <Icon size={16} />
                  </a>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Experience Timeline Section */}
        <div id="experience" className="scroll-mt-24 mt-28 border-t border-slate-200 pt-20">
          <motion.div
            initial={reduced ? false : { opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.6 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="flex flex-col items-center text-center"
          >
            <h3 className="font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
              Professional Experience
            </h3>
            <p className="mt-3 text-sm text-slate-600 max-w-xl">
              {timeline.subtitle}
            </p>
          </motion.div>

          <Timeline reduced={reduced} />

          {/* Recognition, Certifications, and Education */}
          <motion.div
            initial={reduced ? false : { opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-24 grid gap-12 lg:grid-cols-2 lg:gap-16 border-t border-slate-200 pt-16"
          >
            <div>
              <h4 className="font-display text-xl font-bold text-slate-900">
                Key Recognition & Awards
              </h4>
              <ul className="mt-6 divide-y divide-slate-100 rounded-2xl border border-slate-200/90 bg-white p-6 shadow-xs">
                {recognition.awards.map((a) => (
                  <li key={a.name} className="flex items-center justify-between gap-4 py-3.5 first:pt-0 last:pb-0">
                    <span className="font-medium text-sm text-slate-900">{a.name}</span>
                    <span className="shrink-0 font-mono text-xs font-semibold text-slate-500 uppercase">{a.by}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="font-display text-xl font-bold text-slate-900">
                Confirmed Certifications
              </h4>
              <ul className="mt-6 divide-y divide-slate-100 rounded-2xl border border-slate-200/90 bg-white p-6 shadow-xs">
                {recognition.certifications.map((c) => (
                  <li key={c.name} className="flex items-center justify-between gap-4 py-3.5 first:pt-0 last:pb-0">
                    <span className="font-medium text-sm text-slate-900">{c.name}</span>
                    <span className="shrink-0 font-mono text-xs font-semibold text-slate-500 uppercase">{c.meta}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-8 rounded-2xl border border-slate-200/90 bg-white p-6 shadow-xs">
                <span className="font-mono text-[11px] font-medium tracking-wider text-slate-500 uppercase">
                  Education
                </span>
                <p className="mt-2 text-sm font-semibold text-slate-900">
                  {recognition.education}
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function Timeline({ reduced }: { reduced: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.75", "end 0.55"] });
  const scaleY = useSpring(scrollYProgress, { stiffness: 90, damping: 24 });

  return (
    <div ref={ref} className="relative mt-14">
      {/* Timeline center line on desktop, left line on mobile */}
      <span aria-hidden className="absolute top-0 bottom-0 left-4 w-px bg-slate-200 lg:left-1/2 -translate-x-1/2" />
      {/* Lit line drawing in as the timeline scrolls into view */}
      <motion.span
        aria-hidden
        style={{ scaleY: reduced ? 1 : scaleY }}
        className="absolute top-0 bottom-0 left-4 w-px origin-top bg-gradient-to-b from-deep to-violet lg:left-1/2 -translate-x-1/2"
      />

      <div className="space-y-10">
        {experience.map((job, i) => {
          const right = i % 2 === 1;
          return (
            <motion.div
              key={`${job.org}-${job.period}`}
              initial={reduced ? false : { opacity: 0, x: right ? 24 : -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className={`relative flex flex-col lg:flex-row items-center ${
                right ? "lg:flex-row-reverse" : ""
              }`}
            >
              {/* Timeline dot */}
              <span
                aria-hidden
                className="absolute top-6 left-4 size-3 -translate-x-1/2 rounded-full border-2 border-white bg-deep shadow-xs lg:left-1/2"
              />

              {/* Content card */}
              <div className="w-full pl-10 lg:w-1/2 lg:pl-0">
                <article
                  className={`rounded-2xl border border-slate-200/90 bg-white p-7 shadow-xs transition duration-200 hover:border-deep/30 hover:shadow-sm ${
                    right ? "lg:ml-10" : "lg:mr-10"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-xs font-semibold tracking-wider text-deep uppercase">
                      {job.period}
                    </span>
                    <span className="text-xs font-medium text-slate-500">
                      {job.org}
                    </span>
                  </div>

                  <h4 className="font-display mt-2 text-xl font-bold text-slate-900">
                    {job.role}
                  </h4>

                  <p className="mt-3 text-sm leading-relaxed text-slate-600">
                    {job.blurb}
                  </p>

                  <div className="mt-5 pt-3 border-t border-slate-100">
                    <p className="font-mono text-[11px] font-medium text-slate-500">
                      {job.stack}
                    </p>
                  </div>
                </article>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
