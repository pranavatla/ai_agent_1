"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useMotionValue, useSpring, useTransform, type MotionValue } from "motion/react";
import { services } from "@/lib/site";
import { useIsLg, useMedia, useReducedMotion } from "@/lib/media";

const COUNT = services.length;
const HOLD_MS = 1500;
// Circular distance from the active item, in [-COUNT/2, COUNT/2). The index only ever counts up;
// an item leaving the top re-enters at the bottom while it is invisible, so the cycle never rewinds.
const wrap = (d: number) => ((((d + COUNT / 2) % COUNT) + COUNT) % COUNT) - COUNT / 2;

const LEAD = "font-sans text-[20px] font-bold sm:text-3xl md:text-4xl lg:text-[40px] xl:text-5xl";

// Items that have wrapped out of sight can't be clicked or focused.
const useShown = (d: MotionValue<number>) => useTransform(d, (v) => (Math.abs(v) > 2.5 ? "hidden" : "visible"));

function Phrase({ i, idx, k, active, onSelect }: { i: number; idx: MotionValue<number>; k: number; active: boolean; onSelect: (i: number) => void }) {
  const d = useTransform(idx, (v) => wrap(i - v));
  const visibility = useShown(d);
  const y = useTransform(d, (v) => `${v * 1.2}em`);
  const opacity = useTransform(d, [-3, -2.6, -2, -1, 0, 1, 2, 2.6, 3], [0, 0, 0.08, 0.3, 1, 0.3, 0.08, 0, 0]);
  const scale = useTransform(d, [-1, 0, 1], [0.9, 1, 0.9]);
  // Every offset stays positive, or phrases slide back under the lead.
  const x = useTransform(d, [-2, -1, 0, 1, 2], [0, 35 * k, 85 * k, 35 * k, 0]);
  const s = services[i];
  return (
    <motion.li style={{ y, opacity, scale, x, visibility, ...tone(s.color) }} className="svc absolute inset-x-0 top-0 flex h-[1.2em] origin-left items-center whitespace-nowrap text-[var(--svc)]">
      <button
        type="button"
        onClick={() => onSelect(i)}
        aria-current={active || undefined}
        className="cursor-pointer rounded-lg transition-opacity hover:opacity-80 focus-visible:outline-offset-4"
      >
        {s.phrase}
      </button>
    </motion.li>
  );
}

// Sets --c for the .svc class, which lightens the colour on the dark theme (see globals.css).
const tone = (c: string) => ({ "--c": c }) as React.CSSProperties;

// The tile behind each phrase: its colour, its icon and the concrete tools.
// The card-flip overlay copies this tile, so it must stay self-contained (inline --c, no context).
export function ServiceTile({ s }: { s: (typeof services)[number] }) {
  const Icon = s.icon;
  return (
    <div
      className="svc flex h-full w-full flex-col justify-between bg-surface p-6 text-left"
      style={{ ...tone(s.color), backgroundImage: "radial-gradient(120% 90% at 0% 0%, color-mix(in srgb, var(--svc) 18%, transparent), transparent 62%), linear-gradient(160deg, var(--surface) 40%, color-mix(in srgb, var(--svc) 9%, var(--surface)))" }}
    >
      <span className="grid size-12 place-items-center rounded-2xl bg-[var(--c)] text-white shadow-[0_8px_18px_-6px_rgba(15,23,42,0.4)]">
        <Icon aria-hidden className="size-6" />
      </span>
      <div>
        <p className="font-display text-[22px] leading-tight font-bold tracking-[-0.02em] text-[var(--svc)]">
          {s.phrase.replace(/\.$/, "")}
        </p>
        <p className="mt-2 font-mono text-[11px] leading-relaxed text-slate-600">{s.tools}</p>
      </div>
    </div>
  );
}

function Card({ i, idx, active, onSelect }: { i: number; idx: MotionValue<number>; active: boolean; onSelect: (i: number) => void }) {
  const d = useTransform(idx, (v) => wrap(i - v));
  const visibility = useShown(d);
  const y = useTransform(d, (v) => v * 320);
  const x = useTransform(d, [-2, -1, 0, 1, 2], [110, 65, 0, 65, 110]);
  const scale = useTransform(d, [-2, -1, 0, 1, 2], [0.4, 0.52, 1.12, 0.52, 0.4]);
  const opacity = useTransform(d, [-3, -2.6, -2, -1, 0, 1, 2, 2.6, 3], [0, 0, 0.25, 0.8, 1, 0.8, 0.25, 0, 0]);
  return (
    <li className="pointer-events-none absolute inset-x-0 top-0 grid h-[320px] place-items-center">
      {/* Clickable like its phrase; the phrase buttons are the keyboard / screen-reader control. */}
      <motion.div
        onClick={() => onSelect(i)}
        style={{ x, y, scale, opacity, visibility }}
        // Every tile can hand off to the About portrait; the one in front at the time does.
        data-handoff="card"
        data-active={active || undefined}
        aria-hidden
        className="pointer-events-auto relative size-[270px] cursor-pointer overflow-hidden rounded-3xl border border-slate-900/80 shadow-[0_18px_40px_rgba(15,23,42,0.22)]"
      >
        <ServiceTile s={services[i]} />
      </motion.div>
    </li>
  );
}

export default function Services() {
  const reduced = useReducedMotion();
  const lg = useIsLg();
  const md = useMedia("(min-width: 768px)");
  const ref = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const target = useMotionValue(0);
  const idx = useSpring(target, { stiffness: 140, damping: 22 });
  const [step, setStep] = useState(0);

  const lastMove = useRef(0);
  const active = ((step % COUNT) + COUNT) % COUNT;
  const move = (delta: number) => {
    target.set(target.get() + delta);
    setStep((n) => n + delta);
    lastMove.current = performance.now();
  };
  // Clicking an item goes the short way round to it and gives it a full hold before the cycle resumes.
  const select = (i: number) => {
    const delta = Math.round(wrap(i - active));
    if (delta) move(delta);
    else lastMove.current = performance.now();
  };

  // Advance every 1.5s while the section is on screen, pausing while the card flips into About.
  useEffect(() => {
    if (reduced) return;
    let visible = false;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting), { threshold: 0.4 });
    if (ref.current) io.observe(ref.current);
    lastMove.current = performance.now();
    const tick = setInterval(() => {
      if (!visible || stage.current?.dataset.flying) return void (lastMove.current = performance.now());
      if (performance.now() - lastMove.current < HOLD_MS) return;
      target.set(target.get() + 1);
      setStep((n) => n + 1);
      lastMove.current = performance.now();
    }, 100);
    return () => {
      io.disconnect();
      clearInterval(tick);
    };
  }, [reduced, target]);

  if (reduced) {
    return (
      <section id="services" data-covers-galaxy className="light-grid px-6 py-28 text-slate-900">
        <div className={`mx-auto max-w-4xl ${LEAD} font-display tracking-[-0.02em]`}>
          <h2 className="sr-only">What I do</h2>
          <p className="font-sans text-slate-900">I can</p>
          <ul className="mt-4 space-y-3">
            {services.map((s) => (
              <li key={s.phrase} className="svc text-[var(--svc)]" style={tone(s.color)}>
                {s.phrase}
              </li>
            ))}
          </ul>
        </div>
      </section>
    );
  }

  return (
    <section id="services" ref={ref} data-covers-galaxy className="relative">
      <div ref={stage} data-handoff-fade className="light-grid flex h-svh min-h-[560px] items-center overflow-hidden text-slate-900">
        <h2 className="sr-only">What I do</h2>
        <div className="mx-auto flex w-full max-w-7xl items-center gap-8 px-6 sm:px-10 lg:pr-24">
          <div className={`flex min-w-0 flex-1 items-center gap-3 sm:gap-5 ${LEAD}`}>
            <p className="shrink-0 leading-none">I can</p>
            <div className="font-display relative h-[1.2em] flex-1 tracking-[-0.02em]">
              <ul aria-live="off" className="absolute inset-x-0 top-0">
                {services.map((s, i) => (
                  <Phrase key={s.phrase} i={i} idx={idx} k={lg ? 1 : md ? 0.7 : 0.4} active={i === active} onSelect={select} />
                ))}
              </ul>
            </div>
          </div>
          {lg && (
            <div className="relative h-[320px] w-[340px] shrink-0">
              <ul className="absolute inset-x-0 top-0">
                {services.map((s, i) => (
                  <Card key={s.phrase} i={i} idx={idx} active={i === active} onSelect={select} />
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
