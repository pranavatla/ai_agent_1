"use client";

import { useRef } from "react";
import Image from "next/image";
import { motion, useScroll, useSpring, useTransform, type MotionValue } from "motion/react";
import { services } from "@/lib/site";
import { useIsLg, useMedia, useReducedMotion } from "@/lib/media";

const COUNT = services.length;
const DWELL = 0.055;

// Stepped mapping: each index is held across a small band of scroll, with a quick hand-off between.
const input: number[] = [];
const output: number[] = [];
services.forEach((_, i) => {
  const c = i / (COUNT - 1);
  input.push(Math.max(0, c - DWELL), Math.min(1, c + DWELL));
  output.push(i, i);
});

const LEAD = "font-sans text-[20px] font-bold sm:text-3xl md:text-4xl lg:text-[40px] xl:text-5xl";

function Phrase({ i, idx, k }: { i: number; idx: MotionValue<number>; k: number }) {
  const d = useTransform(idx, (v) => i - v);
  const opacity = useTransform(d, [-2, -1, 0, 1, 2], [0.08, 0.3, 1, 0.3, 0.08]);
  const scale = useTransform(d, [-1, 0, 1], [0.9, 1, 0.9]);
  // Every offset stays positive, or phrases slide back under the lead.
  const x = useTransform(d, [-2, -1, 0, 1, 2], [0, 35 * k, 85 * k, 35 * k, 0]);
  const s = services[i];
  return (
    <motion.li style={{ opacity, scale, x, color: s.color }} className="flex h-[1.2em] origin-left items-center whitespace-nowrap">
      {s.phrase}
    </motion.li>
  );
}

function Card({ i, idx }: { i: number; idx: MotionValue<number> }) {
  const d = useTransform(idx, (v) => i - v);
  const x = useTransform(d, [-2, -1, 0, 1, 2], [110, 65, 0, 65, 110]);
  const scale = useTransform(d, [-2, -1, 0, 1, 2], [0.4, 0.52, 1.12, 0.52, 0.4]);
  const opacity = useTransform(d, [-2, -1, 0, 1, 2], [0.25, 0.8, 1, 0.8, 0.25]);
  return (
    <li className="grid h-[320px] place-items-center">
      <motion.div
        style={{ x, scale, opacity }}
        data-handoff={i === COUNT - 1 ? "card" : undefined}
        className="relative size-[270px] overflow-hidden rounded-3xl border border-black/80 shadow-[0_18px_40px_rgba(15,23,42,0.22)]"
      >
        <Image src={services[i].image} alt="" fill sizes="310px" className="object-cover" />
      </motion.div>
    </li>
  );
}

export default function Services() {
  const reduced = useReducedMotion();
  const lg = useIsLg();
  const md = useMedia("(min-width: 768px)");
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const idx = useSpring(useTransform(scrollYProgress, input, output), { stiffness: 260, damping: 32 });
  const y = useTransform(idx, (v) => `${-(v / COUNT) * 100}%`);

  if (reduced) {
    return (
      <section id="services" data-covers-galaxy className="light-grid px-6 py-28 text-slate-900">
        <div className={`mx-auto max-w-4xl ${LEAD} font-display tracking-[-0.02em]`}>
          <p className="font-sans text-slate-900">I can</p>
          <ul className="mt-4 space-y-3">
            {services.map((s) => (
              <li key={s.phrase} style={{ color: s.color }}>
                {s.phrase}
              </li>
            ))}
          </ul>
        </div>
      </section>
    );
  }

  return (
    <section id="services" ref={ref} data-covers-galaxy className="relative h-[320vh]">
      <div data-handoff-fade className="light-grid sticky top-0 flex h-svh items-center overflow-hidden text-slate-900">
        <p className="absolute top-8 left-6 font-mono text-xs tracking-[0.2em] text-slate-500 uppercase sm:left-10">
          What I do
        </p>
        <div className="mx-auto flex w-full max-w-7xl items-center gap-8 px-6 sm:px-10 lg:pr-24">
          <div className={`flex min-w-0 flex-1 items-center gap-3 sm:gap-5 ${LEAD}`}>
            <p className="shrink-0 leading-none">I can</p>
            <div className="font-display relative h-[1.2em] flex-1 tracking-[-0.02em]">
              <motion.ul style={{ y }} className="absolute inset-x-0 top-0">
                {services.map((s, i) => (
                  <Phrase key={s.phrase} i={i} idx={idx} k={lg ? 1 : md ? 0.7 : 0.4} />
                ))}
              </motion.ul>
            </div>
          </div>
          {lg && (
            <div className="relative h-[320px] w-[340px] shrink-0">
              <motion.ul style={{ y }} className="absolute inset-x-0 top-0">
                {services.map((s, i) => (
                  <Card key={s.phrase} i={i} idx={idx} />
                ))}
              </motion.ul>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
