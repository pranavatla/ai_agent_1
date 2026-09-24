"use client";

import { useState, type FormEvent } from "react";
import { EnvelopeSimple, FileArrowDown, MapPin, Phone } from "@phosphor-icons/react/dist/ssr";
import { contact, resumes, site } from "@/lib/site";

const field =
  "mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 placeholder:text-slate-400 transition focus:border-deep focus:bg-surface focus:ring-2 focus:ring-deep/20 focus:outline-none";
const label = "text-sm font-medium text-slate-700";

function Detail({ icon: Icon, title, children, wide }: { icon: typeof Phone; title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`flex items-start gap-4 ${wide ? "sm:col-span-2" : ""}`}>
      <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-deep/10 text-deep">
        <Icon aria-hidden className="size-5" />
      </span>
      <div className="min-w-0">
        <p className={label}>{title}</p>
        <div className="mt-1 text-[15px] font-medium break-words text-slate-900">{children}</div>
      </div>
    </div>
  );
}

export default function Contact() {
  const [status, setStatus] = useState("");

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const body = `${f.get("message")}\n\nFrom: ${f.get("name")}\n${f.get("email")}${f.get("phone") ? `\n${f.get("phone")}` : ""}`;
    // No backend yet: hand the message to the visitor's mail client.
    // To use a real endpoint, replace the next line with: await fetch("/api/contact", { method: "POST", body: f })
    window.location.href = `mailto:${site.email}?subject=${encodeURIComponent(`Hello from ${f.get("name")}`)}&body=${encodeURIComponent(body)}`;
    setStatus("Opening your email app. Review and send your message there.");
  };

  return (
    <section id="contact" data-covers-galaxy className="bg-panel px-4 py-24 text-slate-900 sm:px-10 md:pr-24 lg:py-32">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-12 rounded-[2rem] bg-surface p-6 shadow-[0_24px_60px_rgba(15,23,42,0.08)] sm:p-10 lg:grid-cols-2 lg:gap-16 lg:p-14">
          <div>
            <h2 className="text-4xl font-bold tracking-[-0.03em] sm:text-5xl">Get in touch</h2>
            <p className="mt-5 max-w-md leading-relaxed text-slate-600">
              {contact.lead}
            </p>
            <div className="mt-10 grid gap-x-4 gap-y-6 sm:grid-cols-[1.35fr_1fr]">
              <Detail icon={EnvelopeSimple} title="Email">
                <a href={`mailto:${site.email}`} className="hit hover:text-deep">{site.email}</a>
              </Detail>
              <Detail icon={Phone} title="Phone">
                <a href={`tel:${site.phone.replace(/\s/g, "")}`} className="hit hover:text-deep">{site.phone}</a>
              </Detail>
              <Detail icon={MapPin} title="Location" wide>
                {site.address}
              </Detail>
            </div>
            <div id="resume" className="mt-10 scroll-mt-24">
              <h3 className={label}>Choose a résumé</h3>
              <ul className="mt-3 flex flex-wrap gap-2">
                {resumes.map((r) => (
                  <li key={r.href}>
                    <a
                      href={r.href}
                      download
                      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-800 transition duration-200 ease-snappy hover:border-deep/40 hover:bg-surface hover:text-deep active:scale-[0.97]"
                    >
                      <FileArrowDown aria-hidden className="size-4 text-deep" />
                      {r.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <form onSubmit={onSubmit} className="grid gap-5 sm:grid-cols-2">
            <div>
              <label htmlFor="c-name" className={label}>Name</label>
              <input id="c-name" name="name" type="text" autoComplete="name" required className={field} />
            </div>
            <div>
              <label htmlFor="c-email" className={label}>Email</label>
              <input id="c-email" name="email" type="email" autoComplete="email" required className={field} />
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="c-phone" className={label}>Phone <span className="font-normal text-slate-500">(optional)</span></label>
              <input id="c-phone" name="phone" type="tel" autoComplete="tel" className={field} />
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="c-message" className={label}>Message</label>
              <textarea id="c-message" name="message" rows={5} required className={`${field} resize-y`} />
            </div>
            <button
              type="submit"
              className="rounded-full bg-ink px-6 py-4 text-sm font-semibold text-ground transition duration-200 ease-snappy hover:bg-deep hover:text-white focus-visible:outline-deep active:scale-[0.97] sm:col-span-2"
            >
              Open email draft
            </button>
            <p role="status" aria-live="polite" className="font-mono text-xs text-slate-500 sm:col-span-2">
              {status}
            </p>
          </form>
        </div>
      </div>
    </section>
  );
}
