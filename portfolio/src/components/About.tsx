"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { motion, useMotionValue, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { about, experience, site, socials, timeline } from "@/lib/site";
import { useIsLg, useReducedMotion } from "@/lib/media";

const words = about.bio.split(" ");
// Overlapping slices (2.2x each word's step) so the highlight reads as a wave, mapped across the middle of the pin.
const STEP = 1 / (words.length + 1.2);
const [SWEEP_FROM, SWEEP_TO] = [0.18, 0.72];

function Word({ w, i, progress }: { w: string; i: number; progress: MotionValue<number> }) {
  const a = SWEEP_FROM + i * STEP * (SWEEP_TO - SWEEP_FROM);
  const b = a + 2.2 * STEP * (SWEEP_TO - SWEEP_FROM);
  const color = useTransform(progress, [a, b], ["rgba(15,23,42,0.16)", "rgba(15,23,42,0.9)"]);
  return <motion.span style={{ color }}>{w} </motion.span>;
}

const rise = (delay = 0) => ({
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.4 },
  transition: { duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] as const },
});

export default function About() {
  const reduced = useReducedMotion();
  const lg = useIsLg();
  const stage = useRef<HTMLDivElement>(null);
  const bio = useRef<HTMLParagraphElement>(null);
  const pinned = useScroll({ target: stage, offset: ["start start", "end end"] }).scrollYProgress;
  const flowing = useScroll({ target: bio, offset: ["start 0.9", "end 0.45"] }).scrollYProgress;

  // Pointer parallax: the portrait leans toward the cursor, the galaxy drifts the other way.
  const px = useMotionValue(0);
  const py = useMotionValue(0);
  const sx = useSpring(px, { stiffness: 60, damping: 18 });
  const sy = useSpring(py, { stiffness: 60, damping: 18 });
  const leanX = useTransform(sx, (v) => v * 14);
  const leanY = useTransform(sy, (v) => v * 8);
  const driftX = useTransform(sx, (v) => v * -24);
  const driftY = useTransform(sy, (v) => v * -14);

  useEffect(() => {
    if (reduced) return;
    const onMove = (e: PointerEvent) => {
      px.set((e.clientX / window.innerWidth) * 2 - 1);
      py.set((e.clientY / window.innerHeight) * 2 - 1);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, [reduced, px, py]);

  const pin = !reduced && lg;
  const progress = pin ? pinned : flowing;

  return (
    <section id="about" data-covers-galaxy className="relative">
      {/* Part A: pinned hero. The clip is on the sticky frame, never on this scrolling wrapper. */}
      <div ref={stage} className={pin ? "h-[220vh]" : ""}>
        <div className={`relative flex flex-col overflow-hidden px-6 py-8 sm:px-10 ${pin ? "sticky top-0 h-svh lg:pr-24" : "min-h-svh"}`}>
          <div className="relative z-10 flex items-center justify-between">
            <motion.p {...rise()} className="font-display text-xl font-bold tracking-[0.08em] uppercase">
              {site.brand}
            </motion.p>
            <p className="hidden font-mono text-xs tracking-[0.2em] text-slate-500 uppercase md:block">About</p>
          </div>

          {/* On lg the middle band leaves the flow so the portrait spans the whole frame. */}
          <div className={`flex justify-center ${pin ? "pointer-events-none absolute inset-0 items-end" : "order-2 mt-10 items-end"}`}>
            <div className={`relative aspect-[896/1195] ${pin ? "h-full" : "h-[62svh] max-h-[560px]"}`}>
              <motion.div aria-hidden style={reduced ? undefined : { x: driftX, y: driftY }} className="absolute top-[44%] left-1/2 aspect-square h-[84%] -translate-x-1/2 -translate-y-1/2">
                <motion.div
                  className="galaxy h-full w-full rounded-full blur-[64px]"
                  animate={reduced ? undefined : { scale: [1, 1.08] }}
                  transition={{ duration: 11, repeat: Infinity, repeatType: "mirror", ease: "easeInOut" }}
                />
              </motion.div>
              <motion.div style={reduced ? undefined : { x: leanX, y: leanY }} className="absolute inset-0">
                <motion.div
                  data-handoff="portrait"
                  className="relative h-full w-full"
                  initial={reduced ? false : { opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8 }}
                >
                  <Image src={site.portrait} alt={`Portrait of ${site.name}`} fill sizes="(min-width: 1024px) 80vh, 70vw" className="object-cover" />
                </motion.div>
              </motion.div>
            </div>
          </div>

          <div className={`relative z-10 flex flex-1 flex-col gap-10 lg:flex-row lg:items-center lg:justify-between ${pin ? "" : "order-1 mt-10"}`}>
            <h2 className="font-sans text-6xl leading-[1.05] font-bold tracking-[-0.03em] sm:text-7xl lg:order-2 lg:text-right xl:text-8xl">
              {about.statement.map((line, i) => (
                <motion.span
                  key={line}
                  {...rise(0.12 * i)}
                  className={`block ${i === about.statement.length - 1 ? "bg-gradient-to-r from-deep to-violet bg-clip-text text-transparent" : ""}`}
                >
                  {line}
                </motion.span>
              ))}
            </h2>
            <div className="max-w-[20rem] lg:order-1">
              <p ref={bio} className="text-lg leading-relaxed">
                {reduced
                  ? about.bio
                  : words.map((w, i) => <Word key={i} w={w} i={i} progress={progress} />)}
              </p>
              <a
                href={site.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-6 inline-block font-mono text-xs tracking-[0.2em] text-deep uppercase underline underline-offset-8 hover:text-violet"
              >
                Read more
              </a>
            </div>
          </div>

          <div className={`relative z-10 mt-10 flex items-end justify-between ${pin ? "" : "order-3"}`}>
            <motion.ul {...rise(0.1)} className="flex gap-2">
              {socials.map(({ label, href, icon: Icon }) => (
                <li key={label}>
                  <a
                    href={href}
                    target={href.startsWith("http") ? "_blank" : undefined}
                    rel="noopener noreferrer"
                    aria-label={label}
                    className="grid size-11 place-items-center rounded-xl border border-slate-200 bg-white/70 text-slate-700 shadow-sm backdrop-blur-md transition hover:border-deep/50 hover:text-deep"
                  >
                    <Icon aria-hidden className="size-4" />
                  </a>
                </li>
              ))}
            </motion.ul>
            <motion.p {...rise(0.2)} className="font-mono text-xs tracking-[0.2em] text-slate-500 uppercase">
              {site.address}
            </motion.p>
          </div>
        </div>
      </div>

      <Timeline reduced={reduced} />
    </section>
  );
}

function Timeline({ reduced }: { reduced: boolean }) {
  const ref = useRef<HTMLOListElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 0.7", "end 0.5"] });
  const scaleY = useSpring(scrollYProgress, { stiffness: 90, damping: 24 });

  return (
    <div className="mx-auto max-w-5xl px-6 pt-24 pb-32 sm:px-10">
      <h2 className="font-display text-3xl font-bold tracking-[0.12em] uppercase sm:text-4xl">Experience</h2>
      <p className="mt-3 font-mono text-xs tracking-[0.2em] text-slate-500 uppercase">{timeline.subtitle}</p>
      <ol ref={ref} className="relative mt-16">
        <span aria-hidden className="absolute top-0 bottom-0 left-4 w-px bg-slate-200 lg:left-1/2" />
        <motion.span
          aria-hidden
          style={{ scaleY: reduced ? 1 : scaleY }}
          className="absolute top-0 bottom-0 left-4 w-px origin-top bg-gradient-to-b from-deep to-violet lg:left-1/2"
        />
        {experience.map((job, i) => {
          const right = i % 2 === 1;
          return (
            <li key={job.org} className={`relative pb-12 pl-12 last:pb-0 lg:w-1/2 lg:pl-0 ${right ? "lg:ml-auto lg:pl-14" : "lg:pr-14"}`}>
              <span
                aria-hidden
                className={`absolute top-7 left-4 size-3 -translate-x-1/2 rounded-full bg-deep shadow-[0_0_14px_4px_rgba(29,111,208,0.35)] ${right ? "lg:left-0" : "lg:left-full"}`}
              />
              <motion.article
                initial={reduced ? false : { opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                whileHover={{ y: -4 }}
                className="rounded-2xl border border-slate-200 bg-white/75 p-6 shadow-[0_10px_30px_rgba(15,23,42,0.06)] backdrop-blur-md transition-colors hover:border-deep/40"
              >
                <p className="font-mono text-xs tracking-[0.18em] text-deep uppercase">{job.period}</p>
                <h3 className="mt-3 text-xl font-bold">{job.role}</h3>
                <p className="mt-1 text-slate-500">{job.org}</p>
                <p className="mt-3 text-sm leading-relaxed text-slate-600">{job.blurb}</p>
                <p className="mt-4 font-mono text-[11px] leading-relaxed tracking-[0.06em] text-slate-400">{job.stack}</p>
              </motion.article>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
