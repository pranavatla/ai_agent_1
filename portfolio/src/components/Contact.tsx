"use client";

import { useState, type FormEvent } from "react";
import { EnvelopeSimple, FileArrowDown, MapPin, Phone } from "@phosphor-icons/react";
import { contact, resumes, site } from "@/lib/site";

const field =
  "mt-2 w-full rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 text-slate-900 placeholder:text-slate-400 transition focus:border-deep focus:bg-white focus:ring-2 focus:ring-deep/20 focus:outline-none text-sm";
const label = "font-mono text-xs font-medium tracking-wider text-slate-500 uppercase";

export default function Contact() {
  const [status, setStatus] = useState("");

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const body = `${f.get("message")}\n\nFrom: ${f.get("name")}\nEmail: ${f.get("email")}${
      f.get("phone") ? `\nPhone: ${f.get("phone")}` : ""
    }`;
    window.location.href = `mailto:${site.email}?subject=${encodeURIComponent(
      `Hello from ${f.get("name")}`
    )}&body=${encodeURIComponent(body)}`;
    setStatus("Opening your email app. Review and send your message there.");
  };

  return (
    <section id="contact" className="scroll-mt-20 bg-slate-50/60 px-6 py-24 sm:px-10 border-t border-slate-200/70">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-12 rounded-2xl border border-slate-200/90 bg-white p-8 shadow-xs sm:p-12 lg:grid-cols-2 lg:gap-16">
          {/* Left Column: Direct Contact & Résumé Downloads */}
          <div className="flex flex-col justify-between">
            <div>
              <span className="font-mono text-xs font-medium tracking-[0.2em] text-deep uppercase">
                Contact
              </span>
              <h2 className="font-display mt-3 text-3xl font-extrabold tracking-tight text-slate-900 sm:text-4xl">
                Get in touch
              </h2>
              <p className="mt-4 leading-relaxed text-slate-600">
                {contact.lead}
              </p>

              <div className="mt-8 space-y-4">
                <div className="flex items-center gap-3">
                  <div className="grid size-10 place-items-center rounded-xl bg-deep/10 text-deep">
                    <EnvelopeSimple size={18} />
                  </div>
                  <div>
                    <p className="font-mono text-[11px] text-slate-500 uppercase">Email</p>
                    <a href={`mailto:${site.email}`} className="text-sm font-semibold text-slate-900 hover:text-deep transition">
                      {site.email}
                    </a>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="grid size-10 place-items-center rounded-xl bg-deep/10 text-deep">
                    <Phone size={18} />
                  </div>
                  <div>
                    <p className="font-mono text-[11px] text-slate-500 uppercase">Phone</p>
                    <a href={`tel:${site.phone.replace(/\s/g, "")}`} className="text-sm font-semibold text-slate-900 hover:text-deep transition">
                      {site.phone}
                    </a>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="grid size-10 place-items-center rounded-xl bg-deep/10 text-deep">
                    <MapPin size={18} />
                  </div>
                  <div>
                    <p className="font-mono text-[11px] text-slate-500 uppercase">Location</p>
                    <p className="text-sm font-semibold text-slate-900">{site.address}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Targeted Résumé Downloads */}
            <div id="resume" className="mt-10 scroll-mt-24 border-t border-slate-100 pt-8">
              <h3 className={label}>Tailored Résumés</h3>
              <p className="mt-1 text-xs text-slate-500">Download specific role profiles in PDF format:</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {resumes.map((r) => (
                  <a
                    key={r.href}
                    href={r.href}
                    download
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2 text-xs font-semibold text-slate-800 transition hover:border-deep/40 hover:bg-white hover:text-deep active:scale-95"
                  >
                    <FileArrowDown size={14} className="text-deep" />
                    <span>{r.label}</span>
                  </a>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Contact Message Form */}
          <form onSubmit={onSubmit} className="flex flex-col justify-between rounded-xl bg-slate-50/70 p-6 sm:p-8 border border-slate-200/80">
            <div className="space-y-4">
              <div>
                <label htmlFor="c-name" className={label}>Name</label>
                <input id="c-name" name="name" type="text" autoComplete="name" required className={field} />
              </div>

              <div>
                <label htmlFor="c-email" className={label}>Email</label>
                <input id="c-email" name="email" type="email" autoComplete="email" required className={field} />
              </div>

              <div>
                <label htmlFor="c-phone" className={label}>Phone (optional)</label>
                <input id="c-phone" name="phone" type="tel" autoComplete="tel" className={field} />
              </div>

              <div>
                <label htmlFor="c-message" className={label}>Message</label>
                <textarea id="c-message" name="message" rows={4} required className={`${field} resize-y`} />
              </div>
            </div>

            <div className="mt-6">
              <button
                type="submit"
                className="w-full rounded-xl bg-slate-900 py-3.5 px-6 font-display text-sm font-semibold text-white shadow-xs transition hover:bg-deep active:scale-[0.98]"
              >
                Send Message via Email
              </button>

              {status && (
                <p role="status" aria-live="polite" className="mt-3 text-center font-mono text-xs text-slate-600">
                  {status}
                </p>
              )}
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}
