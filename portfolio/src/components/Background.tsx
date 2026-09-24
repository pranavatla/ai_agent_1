"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { useReducedMotion } from "@/lib/media";

const BAND =
  "radial-gradient(ellipse 120% 70% at 50% 5%, #f7f8fc 0%, #f7f8fc 44%, #eef3ff 48%, #dce7ff 53%, #cbd7ff 59%, #d8cbf6 62%, #edc9e2 66%, #f8dcc6 74%, #fbece3 83%, #f7f8fc 93%)";

export default function Background() {
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll();
  const p = useSpring(scrollYProgress, { stiffness: 70, damping: 24, mass: 0.6 });
  const y = useTransform(p, [0, 1], ["0%", "-22%"]);
  const scale = useTransform(p, [0, 1], [1, 1.12]);
  const [covered, setCovered] = useState(false);

  // Fade the band while a light section, About or the footer holds the viewport middle.
  useEffect(() => {
    const covering = new Set<Element>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) covering.add(e.target);
          else covering.delete(e.target);
        }
        setCovered(covering.size > 0);
      },
      { rootMargin: "-45% 0px -45% 0px" },
    );
    document.querySelectorAll("[data-covers-galaxy]").forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  // Three nested layers, one concern each: scroll, ambient loop, state fade.
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <motion.div style={reduced ? undefined : { y, scale }} className="absolute inset-x-0 top-0 h-[190vh] origin-top">
        <motion.div
          className="h-full w-full"
          animate={reduced ? undefined : { transform: ["translateY(1.5%) scale(1)", "translateY(-2.5%) scale(1.06)"] }}
          transition={{ duration: 26, repeat: Infinity, repeatType: "mirror", ease: "easeInOut" }}
        >
          <div
            className="h-full w-full transition-opacity duration-700"
            style={{ background: BAND, opacity: covered ? 0 : 1 }}
          />
        </motion.div>
      </motion.div>
      {!reduced && <Stars />}
    </div>
  );
}

type Star = { hx: number; hy: number; x: number; y: number; px: number; py: number; vx: number; vy: number; r: number; color: string };

const REACH = 190;

function Stars() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const ptr = { x: 0, y: 0, dx: 0, dy: 0, on: false };
    let stars: Star[] = [];
    let w = 0;
    let h = 0;
    let raf = 0;

    const build = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.min(110, Math.round((w * h) / 15000));
      stars = Array.from({ length: n }, () => {
        const mag = Math.pow(Math.random(), 2.6);
        const cls = Math.random();
        const [lo, hi, sat] = cls < 0.72 ? [210, 240, 70] : cls < 0.92 ? [34, 48, 80] : [330, 350, 60];
        const hue = lo + Math.random() * (hi - lo);
        const light = 48 + Math.random() * 18;
        const x = Math.random() * w;
        const y = Math.random() * h;
        return {
          hx: x, hy: y, x, y, px: x, py: y, vx: 0, vy: 0,
          r: 0.45 + mag * 1.9,
          color: `hsla(${hue.toFixed(0)}, ${sat}%, ${light.toFixed(0)}%, ${(0.22 + mag * 0.45).toFixed(3)})`,
        };
      });
      draw();
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "source-over";
      ctx.lineCap = "round";
      for (const s of stars) {
        ctx.fillStyle = ctx.strokeStyle = s.color;
        // Fast stars smear into a short streak instead of a dotted trail.
        if (Math.hypot(s.x - s.px, s.y - s.py) > 0.4) {
          ctx.lineWidth = s.r * 2;
          ctx.beginPath();
          ctx.moveTo(s.px, s.py);
          ctx.lineTo(s.x, s.y);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };

    const step = () => {
      let busy = false;
      for (const s of stars) {
        s.px = s.x;
        s.py = s.y;
        if (ptr.on) {
          const dx = s.x - ptr.x;
          const dy = s.y - ptr.y;
          const d = Math.hypot(dx, dy);
          if (d < REACH && d > 0.01) {
            const f = 1 - d / REACH;
            s.vx += (dx / d) * f * 1.1 + ptr.dx * f * 0.06;
            s.vy += (dy / d) * f * 1.1 + ptr.dy * f * 0.06;
          }
        }
        s.vx = (s.vx + (s.hx - s.x) * 0.02) * 0.86;
        s.vy = (s.vy + (s.hy - s.y) * 0.02) * 0.86;
        s.x += s.vx;
        s.y += s.vy;
        const still = Math.abs(s.vx) < 0.01 && Math.abs(s.vy) < 0.01;
        if (still && Math.abs(s.x - s.hx) < 0.05 && Math.abs(s.y - s.hy) < 0.05) {
          s.x = s.px = s.hx;
          s.y = s.py = s.hy;
          s.vx = s.vy = 0;
        } else if (!still) {
          busy = true;
        }
      }
      ptr.dx = ptr.dy = 0;
      draw();
      // Sleep once every star is still; the next pointer move wakes the loop.
      raf = busy ? requestAnimationFrame(step) : 0;
    };

    const wake = () => {
      if (!raf) raf = requestAnimationFrame(step);
    };
    const onMove = (e: PointerEvent) => {
      if (ptr.on) {
        ptr.dx += e.clientX - ptr.x;
        ptr.dy += e.clientY - ptr.y;
      }
      ptr.x = e.clientX;
      ptr.y = e.clientY;
      ptr.on = true;
      wake();
    };
    const onLeave = () => {
      ptr.on = false;
      wake();
    };

    build();
    window.addEventListener("resize", build);
    window.addEventListener("pointermove", onMove, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeave);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", build);
      window.removeEventListener("pointermove", onMove);
      document.documentElement.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}
