"use client";

import { motion } from "motion/react";
import { services } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

export default function Services() {
  const reduced = useReducedMotion();

  return (
    <section id="services" className="scroll-mt-20 bg-slate-50/60 px-6 py-24 sm:px-10 border-y border-slate-200/70">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduced ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.6 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-center text-center"
        >
          <h2 className="font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Capabilities & Focus
          </h2>
          <p className="mt-3 text-sm text-slate-600 max-w-xl">
            Enterprise cloud operations combined with hands-on AI engineering and automation.
          </p>
        </motion.div>

        <div className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {services.map((service, i) => {
            const Icon = service.icon;
            return (
              <motion.div
                key={service.phrase}
                initial={reduced ? false : { opacity: 0, y: 24, scale: 0.97 }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ duration: 0.55, delay: reduced ? 0 : (i % 3) * 0.09, ease: [0.22, 1, 0.36, 1] }}
                whileHover={reduced ? undefined : { y: -4, transition: { duration: 0.2, delay: 0, ease: "easeOut" } }}
                className="group relative flex flex-col justify-between rounded-2xl border border-slate-200/90 bg-white p-7 shadow-xs transition-colors duration-200 hover:border-deep/30 hover:shadow-md"
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
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
