"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { scroll } from "motion";
import { site } from "@/lib/site";
import { useIsLg, useReducedMotion } from "@/lib/media";

const clamp = (v: number) => Math.min(1, Math.max(0, v));
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// Shared-element flight: the service card in front flips over and lands as the About portrait.
// Both rects are read live every frame, so nothing about either position is hard-coded.
export default function Handoff() {
  const reduced = useReducedMotion();
  const lg = useIsLg();
  const overlay = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const flipper = useRef<HTMLDivElement>(null);
  const faces = useRef<HTMLDivElement[]>([]);
  const tile = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (reduced || !lg) return;
    const dst = document.querySelector<HTMLElement>('[data-handoff="portrait"]');
    const fade = document.querySelector<HTMLElement>("[data-handoff-fade]");
    const about = document.getElementById("about");
    const o = overlay.current!;
    const b = box.current!;
    const f = flipper.current!;
    const t = tile.current!;
    if (!dst || !fade || !about) return;
    // The source is whichever service tile is in front when the flight starts; it stays locked
    // (and the Services cycle stays paused via data-flying) until the page scrolls back above it.
    let src: HTMLElement | null = null;
    let srcRadius = 24;
    let raf = 0;

    const release = () => {
      if (src) src.style.visibility = "";
      src = null;
      delete fade.dataset.flying;
    };

    const frame = () => {
      raf = 0;
      // 0 as About's top enters the fold, 1 as it reaches the top: the 100vh where Services scrolls away.
      const p = clamp(1 - about.getBoundingClientRect().top / window.innerHeight);
      dst.style.visibility = p < 1 ? "hidden" : "";
      // Dissolve Services once the card is past edge-on, so the portrait lands on the starfield.
      fade.style.opacity = String(1 - clamp((p - 0.15) / 0.25));
      if (p <= 0) {
        release();
        o.style.display = "none";
        return;
      }
      if (!src) {
        src = document.querySelector<HTMLElement>('[data-handoff="card"][data-active]');
        if (!src) return;
        fade.dataset.flying = "1";
        // The flying front face is a copy of the tile that was in front.
        t.replaceChildren(src.firstElementChild!.cloneNode(true));
        // The card is scaled by the roller, so its on-screen corner is its CSS radius times that scale.
        srcRadius = parseFloat(getComputedStyle(src).borderTopLeftRadius) || 24;
      }
      src.style.visibility = "hidden";
      const flying = p < 1;
      o.style.display = flying ? "block" : "none";
      if (!flying) return;

      const a = src.getBoundingClientRect();
      const d = dst.getBoundingClientRect();
      const e = ease(p);
      b.style.left = `${lerp(a.left, d.left, e)}px`;
      b.style.top = `${lerp(a.top, d.top, e)}px`;
      b.style.width = `${lerp(a.width, d.width, e)}px`;
      b.style.height = `${lerp(a.height, d.height, e)}px`;
      // The tile is laid out at the card's own size (inside its 1px border) and scaled, exactly as the roller scales the card.
      t.style.width = `${src.clientWidth}px`;
      t.style.height = `${src.clientHeight}px`;
      t.style.transform = `scale(${parseFloat(b.style.width) / src.offsetWidth}, ${parseFloat(b.style.height) / src.offsetHeight})`;
      f.style.transform = `rotateY(${ease(clamp(p / 0.3)) * 180}deg)`;
      const start = srcRadius * (a.width / src.offsetWidth);
      const r = lerp(start, 44, e) * (p < 0.88 ? 1 : 1 - (p - 0.88) / 0.12);
      for (const face of faces.current) face.style.borderRadius = `${r}px`;
      // Keep following while the roller's spring settles, even without scroll events.
      raf = requestAnimationFrame(frame);
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(frame);
    };

    frame();
    // Motion's frame-batched scroll observer, not a raw window scroll listener.
    const stopScroll = scroll(() => schedule());
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(raf);
      stopScroll();
      window.removeEventListener("resize", schedule);
      release();
      dst.style.visibility = "";
      fade.style.opacity = "";
      o.style.display = "none";
    };
  }, [reduced, lg]);

  const face = "absolute inset-0 overflow-hidden [backface-visibility:hidden]";
  return (
    <div ref={overlay} aria-hidden className="pointer-events-none fixed inset-0 z-40 [perspective:1600px]" style={{ display: "none" }}>
      <div ref={box} className="absolute">
        <div ref={flipper} className="relative h-full w-full [transform-style:preserve-3d]">
          <div ref={(el) => { if (el) faces.current[0] = el; }} className={`${face} border border-slate-900/80`}>
            <div ref={tile} className="origin-top-left" />
          </div>
          <div ref={(el) => { if (el) faces.current[1] = el; }} className={face} style={{ transform: "rotateY(180deg)" }}>
            <Image src={site.portrait} alt="" fill sizes="80vh" className="object-cover" />
          </div>
        </div>
      </div>
    </div>
  );
}
