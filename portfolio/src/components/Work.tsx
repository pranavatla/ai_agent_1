"use client";

import Image from "next/image";
import { motion } from "motion/react";
import { ArrowUpRight } from "@phosphor-icons/react";
import { projects, work } from "@/lib/site";
import { useReducedMotion } from "@/lib/media";

export default function Work() {
  const reduced = useReducedMotion();

  return (
    <section id="work" className="scroll-mt-20 px-6 py-24 sm:px-10">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-col items-center text-center">
          <h2 className="font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
            Featured Projects
          </h2>
          <p className="mt-3 text-sm text-slate-600 max-w-xl">
            {work.subtitle}
          </p>
        </div>

        <div className="mt-14 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <article
              key={project.title}
              className="group flex flex-col overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-xs transition duration-200 hover:-translate-y-1 hover:border-deep/30 hover:shadow-md"
            >
              <a
                href={project.href}
                target="_blank"
                rel="noopener noreferrer"
                className="relative block aspect-[16/10] w-full overflow-hidden bg-slate-100"
              >
                <Image
                  src={project.image}
                  alt={project.alt}
                  fill
                  priority
                  sizes="(min-width: 1024px) 33vw, (min-width: 640px) 50vw, 100vw"
                  className="object-cover transition duration-300 group-hover:scale-[1.03]"
                />
              </a>

              <div className="flex flex-1 flex-col justify-between p-6">
                <div>
                  <span className="font-mono text-[11px] font-medium tracking-wider text-deep uppercase">
                    {project.tag}
                  </span>
                  <h3 className="font-display mt-2 text-xl font-bold text-slate-900 group-hover:text-deep transition-colors">
                    {project.title}
                  </h3>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
                  <a
                    href={project.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 transition group-hover:text-deep"
                  >
                    <span>Launch project</span>
                    <ArrowUpRight size={15} weight="bold" />
                  </a>
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
