"use client";

import { useRef } from "react";
import Image from "next/image";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { ArrowUpRight } from "lucide-react";
import { projects, work } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

const CARDS = 10;
// Three real projects cycled around the ring. Tiles are 16:9, so the radius is sized
// to keep each tile narrower than the chord to its neighbour (2R·sin18° ≈ 0.618R).
const ring = Array.from({ length: CARDS }, (_, i) => projects[i % projects.length]);

function Heading() {
  return (
    <div className="text-center">
      <h2 className="font-display text-4xl font-bold tracking-[0.12em] uppercase sm:text-5xl">Work</h2>
      <p className="mt-3 font-mono text-xs tracking-[0.2em] text-slate-500 uppercase">{work.subtitle}</p>
    </div>
  );
}

function Face({ p, back }: { p: (typeof projects)[number]; back?: boolean }) {
  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-2xl border border-slate-200 bg-white text-white shadow-[0_12px_30px_rgba(15,23,42,0.12)] [backface-visibility:hidden]"
      style={back ? { transform: "rotateY(180deg)", filter: "grayscale(0.5)", opacity: 0.45 } : undefined}
    >
      <Image src={p.image} alt={back ? "" : p.alt} fill sizes="(min-width: 1024px) 600px, 320px" className="object-cover" />
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/85 to-transparent p-3 pt-10 text-left md:p-4">
        <p className="font-mono text-[10px] tracking-[0.16em] text-sky-300 uppercase">{p.tag}</p>
        <p className="font-display mt-1 text-sm font-bold tracking-[-0.02em]">{p.title}</p>
      </div>
    </div>
  );
}

export default function Work() {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const rotateY = useSpring(useTransform(scrollYProgress, [0, 1], [0, -360]), { stiffness: 70, damping: 22 });

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
    <section id="work" ref={ref} className="relative h-[340vh]">
      {/* The clip lives on the sticky stage: overflow on the section would stop it pinning. */}
      <div className="sticky top-0 flex h-svh flex-col items-center justify-center gap-10 overflow-hidden">
        <Heading />
        <div className="relative h-[170px] w-full [perspective:1800px] md:h-[230px] lg:h-[300px]">
          <motion.div
            style={{ rotateY, transformStyle: "preserve-3d" }}
            className="absolute inset-0 [--r:340px] md:[--r:470px] lg:[--r:630px]"
          >
            {ring.map((p, i) => (
              <a
                key={i}
                href={p.href}
                target="_blank"
                rel="noopener noreferrer"
                // Only the first lap of projects is reachable by keyboard / screen reader.
                tabIndex={i < projects.length ? undefined : -1}
                aria-hidden={i < projects.length ? undefined : true}
                aria-label={i < projects.length ? `${p.title}, opens in a new tab` : undefined}
                className="group absolute top-1/2 left-1/2 -mt-[58px] -ml-[104px] h-[117px] w-[208px] md:-mt-[81px] md:-ml-[144px] md:h-[162px] md:w-[288px] lg:-mt-[108px] lg:-ml-[192px] lg:h-[216px] lg:w-[384px]"
                style={{ transform: `rotateY(${i * 36}deg) translateZ(var(--r))`, transformStyle: "preserve-3d" }}
              >
                <Face p={p} />
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
