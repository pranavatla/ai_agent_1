"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import type { MotionValue } from "motion/react";
import type { ModelViewerElement } from "@google/model-viewer";
import { site } from "@/lib/site";

export default function Portrait({ progress, reduced }: { progress: MotionValue<number>; reduced: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const container = host.current!;
    let disposed = false;
    let viewer: ModelViewerElement | undefined;
    let stopScroll: (() => void) | undefined;
    const observer = new IntersectionObserver(async ([entry]) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();
      try {
        await import("@google/model-viewer");
        if (disposed) return;
        viewer = document.createElement("model-viewer") as ModelViewerElement;
        const attributes = {
          src: "/media/portrait.glb",
          alt: `Interactive 3D portrait of ${site.name}. Drag left or right, or use arrow keys to rotate.`,
          "camera-controls": "",
          "disable-zoom": "",
          "disable-pan": "",
          "touch-action": "pan-y",
          "camera-orbit": "0deg 90deg 105%",
          "min-camera-orbit": "-55deg 65deg auto",
          "max-camera-orbit": "55deg 105deg auto",
          "field-of-view": "30deg",
          "interaction-prompt": "none",
          "shadow-intensity": "0",
          exposure: "1",
        };
        for (const [name, value] of Object.entries(attributes)) viewer.setAttribute(name, value);
        viewer.style.cssText = "width:100%;height:100%;background:transparent;--progress-bar-height:0;";
        viewer.addEventListener("load", () => { if (!disposed) setLoaded(true); });
        viewer.addEventListener("error", () => { if (!disposed) setLoaded(false); });
        container.append(viewer);
        if (!reduced) {
          const turn = (p: number) => { viewer!.cameraOrbit = `${-18 + p * 36}deg 90deg 105%`; };
          turn(progress.get());
          stopScroll = progress.on("change", turn);
        }
      } catch {
        // Keep the original portrait if WebGL or the viewer cannot load.
      }
    }, { rootMargin: "400px" });
    observer.observe(container);
    return () => {
      disposed = true;
      observer.disconnect();
      stopScroll?.();
      viewer?.remove();
    };
  }, [progress, reduced]);

  return (
    <div className="pointer-events-auto relative h-full w-full">
      <Image src={site.portrait} alt={`Portrait of ${site.name}`} fill sizes="(min-width: 1024px) 80vh, 70vw" className={`pointer-events-none object-cover ${loaded ? "invisible" : ""}`} />
      <div ref={host} className={`absolute inset-0 ${loaded ? "" : "opacity-0"}`} />
      {loaded && <span className="pointer-events-none absolute bottom-5 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-surface/80 px-3 py-1 font-mono text-[10px] text-slate-500">Drag to explore · 3D portrait</span>}
    </div>
  );
}
