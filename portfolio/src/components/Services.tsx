"use client";

import { motion } from "motion/react";
import { services } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

export default function Services() {
  const reduced = useReducedMotion();

  return (
    <section id="services" className="scroll-mt-20 bg-slate-50/60 px-6 py-24 sm:px-10 border-y border-slate-200/70">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-col items-center text-center">
          <h2 className="font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Capabilities & Focus
          </h2>
          <p className="mt-3 text-sm text-slate-600 max-w-xl">
            Enterprise cloud operations combined with hands-on AI engineering and automation.
          </p>
        </div>

        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {services.map((service) => {
            const Icon = service.icon;
            return (
              <div
                key={service.phrase}
                className="group relative flex flex-col justify-between rounded-2xl border border-slate-200/90 bg-white p-7 shadow-xs transition duration-200 hover:-translate-y-1 hover:border-deep/30 hover:shadow-md"
              >
                <div>
                  <div className="grid size-12 place-items-center rounded-xl bg-deep/10 text-deep">
                    <Icon className="size-6" weight="regular" />
                  </div>
                  <h3 className="font-display mt-5 text-xl font-bold text-slate-900 group-hover:text-deep transition-colors">
                    {service.phrase}
                  </h3>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-100">
                  <p className="font-mono text-xs leading-relaxed text-slate-600">
                    {service.tools}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
