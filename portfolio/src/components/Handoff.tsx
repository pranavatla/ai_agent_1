"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { scroll } from "motion";
import { services, site } from "@/lib/site";
import { useIsLg, useReducedMotion } from "@/lib/media";
import { ServiceTile } from "./Services";

const clamp = (v: number) => Math.min(1, Math.max(0, v));
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// Shared-element flight: the last service card flips over and lands as the About portrait.
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
    const src = document.querySelector<HTMLElement>('[data-handoff="card"]');
    const dst = document.querySelector<HTMLElement>('[data-handoff="portrait"]');
    const fade = document.querySelector<HTMLElement>("[data-handoff-fade]");
    const about = document.getElementById("about");
    const o = overlay.current!;
    const b = box.current!;
    const f = flipper.current!;
    const t = tile.current!;
    if (!src || !dst || !fade || !about) return;
    // The card is scaled by the roller, so its on-screen corner is its CSS radius times that scale.
    const srcRadius = parseFloat(getComputedStyle(src).borderTopLeftRadius) || 24;
    let raf = 0;

    const frame = () => {
      raf = 0;
      // 0 as About's top enters the fold, 1 as it reaches the top: exactly the 100vh where Services unpins.
      const p = clamp(1 - about.getBoundingClientRect().top / window.innerHeight);
      src.style.visibility = p > 0 ? "hidden" : "";
      dst.style.visibility = p < 1 ? "hidden" : "";
      // Dissolve Services once the card is past edge-on, so the portrait lands on the starfield.
      fade.style.opacity = String(1 - clamp((p - 0.15) / 0.25));
      const flying = p > 0 && p < 1;
      o.style.display = flying ? "block" : "none";
      if (!flying) return;

      const a = src.getBoundingClientRect();
      const d = dst.getBoundingClientRect();
      const e = ease(p);
      b.style.left = `${lerp(a.left, d.left, e)}px`;
      b.style.top = `${lerp(a.top, d.top, e)}px`;
      b.style.width = `${lerp(a.width, d.width, e)}px`;
      b.style.height = `${lerp(a.height, d.height, e)}px`;
      // The tile is laid out at the card's own size and scaled, exactly as the roller scales the card.
      // Inside the 1px border, like the real card's tile; scaled by the card's own scale factor.
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
      src.style.visibility = "";
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
            <div ref={tile} className="origin-top-left">
              <ServiceTile s={services[services.length - 1]} />
            </div>
          </div>
          <div ref={(el) => { if (el) faces.current[1] = el; }} className={face} style={{ transform: "rotateY(180deg)" }}>
            <Image src={site.portrait} alt="" fill sizes="80vh" className="object-cover" />
          </div>
        </div>
      </div>
    </div>
  );
}
