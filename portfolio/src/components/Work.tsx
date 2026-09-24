"use client";

import { useRef } from "react";
import Image from "next/image";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { ArrowUpRight } from "@phosphor-icons/react/dist/ssr";
import { projects, work } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

// Three projects on a ring, 120deg apart. Tiles are 16:9; the radius keeps each tile narrower
// than the chord to its neighbour (2R·sin60 ≈ 1.73R), so they never intersect.
const STEP = 360 / projects.length;
// Each project holds the front for a band of scroll, with a quick turn between (like the Services roller).
const DWELL = 0.12;
const input: number[] = [];
const output: number[] = [];
projects.forEach((_, i) => {
  const c = i / (projects.length - 1);
  input.push(Math.max(0, c - DWELL), Math.min(1, c + DWELL));
  output.push(-i * STEP, -i * STEP);
});

function Heading() {
  return (
    <div className="text-center">
      <h2 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">Work</h2>
      <p className="mt-3 text-base text-slate-600">{work.subtitle}</p>
    </div>
  );
}

function Face({ p, back, play }: { p: (typeof projects)[number]; back?: boolean; play?: boolean }) {
  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-2xl border border-slate-200 bg-surface text-white shadow-[0_12px_30px_rgba(15,23,42,0.12)] [backface-visibility:hidden]"
      style={back ? { transform: "rotateY(180deg)", filter: "grayscale(0.5)", opacity: 0.45 } : undefined}
    >
      {/* Only the front face plays the video; the back shows the still poster. */}
      {p.video && !back && play ? (
        <video
          src={p.video}
          poster={p.image}
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
          aria-label={p.alt}
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <Image src={p.image} alt={back ? "" : p.alt} fill sizes="(min-width: 1024px) 700px, 320px" className="object-cover" />
      )}
      {/* Dark scrim tall enough to keep the caption legible even over a white screenshot. */}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#0b1220]/95 via-[#0b1220]/70 to-transparent p-3 pt-10 text-left md:p-5 md:pt-16">
        <p className="truncate font-mono text-[10px] text-sky-300 md:text-xs">{p.tag}</p>
        <p className="font-display mt-0.5 text-sm font-bold tracking-[-0.02em] md:mt-1 md:text-lg">{p.title}</p>
      </div>
    </div>
  );
}

export default function Work() {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const rotateY = useSpring(useTransform(scrollYProgress, input, output), { stiffness: 70, damping: 22 });

  if (reduced) {
    return (
      <section id="work" className="px-6 py-28">
        <Heading />
        <ul className="mx-auto mt-14 grid max-w-5xl gap-6 sm:grid-cols-3">
          {projects.map((p) => (
            <li key={p.title}>
              <a href={p.href} target="_blank" rel="noopener noreferrer" className="relative block aspect-video">
                <Face p={p} />
              </a>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  return (
    <section id="work" ref={ref} className="relative h-[300vh]">
      {/* The clip lives on the sticky stage: overflow on the section would stop it pinning. */}
      <div className="sticky top-0 flex h-svh flex-col items-center justify-center gap-12 overflow-hidden">
        <Heading />
        <div className="relative h-[190px] w-full [perspective:1800px] md:h-[270px] lg:h-[360px]">
          <motion.div
            style={{ rotateY, transformStyle: "preserve-3d" }}
            className="absolute inset-0 [--r:190px] md:[--r:280px] lg:[--r:380px]"
          >
            {projects.map((p, i) => (
              <a
                key={p.title}
                href={p.href}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`${p.title}, opens in a new tab`}
                className="group absolute top-1/2 left-1/2 -mt-[76px] -ml-[136px] h-[153px] w-[272px] md:-mt-[112px] md:-ml-[200px] md:h-[225px] md:w-[400px] lg:-mt-[153px] lg:-ml-[272px] lg:h-[306px] lg:w-[544px]"
                style={{ transform: `rotateY(${i * STEP}deg) translateZ(var(--r))`, transformStyle: "preserve-3d" }}
              >
                <Face p={p} play />
                <Face p={p} back />
                <ArrowUpRight
                  aria-hidden
                  className="absolute top-3 right-3 size-5 text-white/80 opacity-0 transition group-hover:opacity-100 [backface-visibility:hidden]"
                />
              </a>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
