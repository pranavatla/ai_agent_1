"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { services, site } from "@/lib/site";
import { useIsLg, useReducedMotion } from "@/lib/media";

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

  useEffect(() => {
    if (reduced || !lg) return;
    const src = document.querySelector<HTMLElement>('[data-handoff="card"]');
    const dst = document.querySelector<HTMLElement>('[data-handoff="portrait"]');
    const fade = document.querySelector<HTMLElement>("[data-handoff-fade]");
    const about = document.getElementById("about");
    const o = overlay.current!;
    const b = box.current!;
    const f = flipper.current!;
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
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", schedule);
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
          <div ref={(el) => { if (el) faces.current[0] = el; }} className={`${face} border border-black/80`}>
            <Image src={services[services.length - 1].image} alt="" fill sizes="310px" className="object-cover" />
          </div>
          <div ref={(el) => { if (el) faces.current[1] = el; }} className={face} style={{ transform: "rotateY(180deg)" }}>
            <Image src={site.portrait} alt="" fill sizes="80vh" className="object-cover" />
          </div>
        </div>
      </div>
    </div>
  );
}
