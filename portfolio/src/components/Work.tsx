"use client";

import { useEffect, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import Image from "next/image";
import { motion } from "motion/react";
import { ArrowUpRight, CaretLeft, CaretRight } from "@phosphor-icons/react/dist/ssr";
import { projects, work } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

// Four projects on a ring, 90deg apart. Tiles are 16:9; the radius keeps each tile narrower
// than the chord to its neighbour (2R·sin45 ≈ 1.41R), so they never intersect.
const N = projects.length;
const STEP = 360 / N;
const STILL_MS = 3000; // how long a tile without a video (or whose video can't play) stays in front
const SWIPE_PX = 50;
const mod = (n: number) => ((n % N) + N) % N;

function Heading() {
  return (
    <div className="text-center">
      <h2 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">Work</h2>
      <p className="mt-3 text-base text-slate-600">{work.subtitle}</p>
    </div>
  );
}

function Face({ p, slot, back }: { p: (typeof projects)[number]; slot?: number; back?: boolean }) {
  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-2xl border border-slate-200 bg-surface text-white shadow-[0_12px_30px_rgba(15,23,42,0.12)] [backface-visibility:hidden]"
      style={back ? { transform: "rotateY(180deg)", filter: "grayscale(0.5)", opacity: 0.45 } : undefined}
    >
      {/* Only the front face carries the video (played by the ring, one at a time); the back shows the poster. */}
      {p.video && slot !== undefined ? (
        <video
          data-slot={slot}
          src={p.video}
          poster={p.image}
          muted
          playsInline
          preload="metadata"
          aria-label={p.alt}
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <Image src={p.image} alt={back ? "" : p.alt} fill draggable={false} sizes="(min-width: 1024px) 700px, 320px" className="object-cover" />
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
  const section = useRef<HTMLElement>(null);
  const ring = useRef<HTMLDivElement>(null);
  const [index, setIndex] = useState(0); // counts without wrapping, so the ring always turns the short way round
  const [inView, setInView] = useState(false);
  const [held, setHeld] = useState(false); // hover or keyboard focus pauses auto-rotation
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const wheel = useRef({ acc: 0, last: 0 });
  const front = mod(index);
  const go = (delta: number) => setIndex((i) => i + delta);

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.5 });
    if (section.current) io.observe(section.current);
    return () => io.disconnect();
  }, []);

  // A new front tile starts its video from the beginning.
  useEffect(() => {
    const v = ring.current?.querySelector<HTMLVideoElement>(`video[data-slot="${front}"]`);
    if (v) v.currentTime = 0;
  }, [front]);

  // The front tile plays its video once, then the ring moves on; a tile without a video stays 3s.
  useEffect(() => {
    if (reduced || !ring.current) return;
    const videos = [...ring.current.querySelectorAll<HTMLVideoElement>("video[data-slot]")];
    const v = videos.find((x) => Number(x.dataset.slot) === front);
    for (const x of videos) if (x !== v || !inView) x.pause();
    if (!inView) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const next = () => {
      if (held) v?.play().catch(() => {}); // paused by hover/focus: keep the clip going, don't rotate
      else setIndex((i) => i + 1);
    };
    if (v) {
      v.addEventListener("ended", next);
      v.play().catch(() => (timer = setTimeout(next, STILL_MS))); // e.g. autoplay blocked in low-power mode
    } else if (!held) {
      timer = setTimeout(next, STILL_MS);
    }
    return () => {
      v?.removeEventListener("ended", next);
      clearTimeout(timer);
    };
  }, [front, inView, held, reduced]);

  // Horizontal swipes only: vertical movement is left to the page (touch-action: pan-y).
  const onPointerDown = (e: PointerEvent) => (drag.current = { x: e.clientX, y: e.clientY, moved: false });
  const onPointerUp = (e: PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    if (Math.abs(dx) > SWIPE_PX && Math.abs(dx) > Math.abs(e.clientY - d.y)) {
      d.moved = true;
      go(dx < 0 ? 1 : -1);
    }
  };
  // Trackpad two-finger horizontal swipe; one step per gesture.
  const onWheel = (e: WheelEvent) => {
    if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
    const w = wheel.current;
    w.acc += e.deltaX;
    if (Math.abs(w.acc) > 60 && e.timeStamp - w.last > 700) {
      go(w.acc > 0 ? 1 : -1);
      w.last = e.timeStamp;
      w.acc = 0;
    }
  };

  if (reduced) {
    return (
      <section id="work" className="px-6 py-28">
        <Heading />
        <ul className="mx-auto mt-14 grid max-w-5xl gap-6 sm:grid-cols-2">
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

  const control =
    "grid size-11 place-items-center rounded-full border border-slate-200 bg-surface text-slate-700 shadow-sm transition duration-200 ease-snappy hover:border-deep/40 hover:text-deep active:scale-[0.94]";

  return (
    <section
      id="work"
      ref={section}
      aria-roledescription="carousel"
      aria-label="Projects"
      className="relative flex min-h-svh flex-col items-center justify-center gap-10 overflow-hidden py-20"
    >
      <Heading />
      <div
        className="relative h-[190px] w-full touch-pan-y select-none [perspective:1800px] md:h-[270px] lg:h-[360px]"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={() => (drag.current = null)}
        onWheel={onWheel}
        onPointerEnter={(e) => e.pointerType === "mouse" && setHeld(true)}
        onPointerLeave={(e) => e.pointerType === "mouse" && setHeld(false)}
        onFocus={() => setHeld(true)}
        onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setHeld(false)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") go(1);
          if (e.key === "ArrowLeft") go(-1);
        }}
        // A swipe that started on a card must not also open its link.
        onClickCapture={(e) => {
          if (drag.current?.moved) e.preventDefault();
          drag.current = null;
        }}
      >
        <motion.div
          ref={ring}
          animate={{ rotateY: -index * STEP }}
          transition={{ type: "spring", stiffness: 60, damping: 16 }}
          style={{ transformStyle: "preserve-3d" }}
          className="absolute inset-0 [--r:215px] md:[--r:310px] lg:[--r:420px]"
        >
          {projects.map((p, i) => (
            <a
              key={p.title}
              href={p.href}
              target="_blank"
              rel="noopener noreferrer"
              draggable={false}
              aria-label={`${p.title}, opens in a new tab`}
              // Only the tile in front is reachable by keyboard; the arrows and dots move between them.
              tabIndex={i === front ? undefined : -1}
              aria-hidden={i === front ? undefined : true}
              className="group absolute top-1/2 left-1/2 -mt-[76px] -ml-[136px] h-[153px] w-[272px] md:-mt-[112px] md:-ml-[200px] md:h-[225px] md:w-[400px] lg:-mt-[153px] lg:-ml-[272px] lg:h-[306px] lg:w-[544px]"
              style={{ transform: `rotateY(${i * STEP}deg) translateZ(var(--r))`, transformStyle: "preserve-3d" }}
            >
              <Face p={p} slot={i} />
              <Face p={p} back />
              <ArrowUpRight
                aria-hidden
                className="absolute top-3 right-3 size-5 text-white/80 opacity-0 transition group-hover:opacity-100 [backface-visibility:hidden]"
              />
            </a>
          ))}
        </motion.div>
      </div>
      <div className="flex items-center gap-4">
        <button type="button" aria-label="Previous project" onClick={() => go(-1)} className={control}>
          <CaretLeft aria-hidden className="size-4" />
        </button>
        <div className="flex gap-2">
          {projects.map((p, i) => (
            <button
              key={p.title}
              type="button"
              aria-label={`Show ${p.title}`}
              aria-current={i === front || undefined}
              // Jump the short way round from the current tile.
              onClick={() => go(((i - front + N + N / 2) % N) - N / 2)}
              className="grid size-6 place-items-center"
            >
              <span className={`block h-1.5 rounded-full transition-all duration-300 ${i === front ? "w-5 bg-deep" : "w-1.5 bg-slate-300"}`} />
            </button>
          ))}
        </div>
        <button type="button" aria-label="Next project" onClick={() => go(1)} className={control}>
          <CaretRight aria-hidden className="size-4" />
        </button>
      </div>
    </section>
  );
}
