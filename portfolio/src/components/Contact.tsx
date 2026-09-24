"use client";

import { useState, type FormEvent } from "react";
import { Mail, MapPin, Phone } from "lucide-react";
import { contact, site } from "@/lib/site";

const field =
  "mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 placeholder:text-slate-400 transition focus:border-deep focus:bg-white focus:ring-2 focus:ring-deep/20 focus:outline-none";
const label = "font-mono text-[11px] tracking-[0.16em] text-slate-500 uppercase";

function Detail({ icon: Icon, title, children, wide }: { icon: typeof Mail; title: string; children: React.ReactNode; wide?: boolean }) {
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
    const body = `${f.get("message")}\n\n— ${f.get("name")}\n${f.get("email")}${f.get("phone") ? `\n${f.get("phone")}` : ""}`;
    // No backend yet: hand the message to the visitor's mail client.
    // To use a real endpoint, replace the next line with: await fetch("/api/contact", { method: "POST", body: f })
    window.location.href = `mailto:${site.email}?subject=${encodeURIComponent(`Hello from ${f.get("name")}`)}&body=${encodeURIComponent(body)}`;
    setStatus("Opening your mail app…");
  };

  return (
    <section id="contact" data-covers-galaxy className="bg-panel px-4 py-24 text-slate-900 sm:px-10 md:pr-24 lg:py-32">
      <div className="mx-auto max-w-6xl">
        <h2 className="font-display text-sm font-bold tracking-[0.2em] text-deep uppercase">Contact</h2>
        <div className="mt-8 grid gap-12 rounded-[2rem] bg-white p-6 shadow-[0_24px_60px_rgba(15,23,42,0.08)] sm:p-10 lg:grid-cols-2 lg:gap-16 lg:p-14">
          <div>
            <p className="text-4xl font-bold tracking-[-0.03em] sm:text-5xl">Get in touch</p>
            <p className="mt-5 max-w-md leading-relaxed text-slate-600">
              {contact.lead}
            </p>
            <div className="mt-10 grid gap-x-4 gap-y-6 sm:grid-cols-[1.35fr_1fr]">
              <Detail icon={Mail} title="Email">
                <a href={`mailto:${site.email}`} className="hover:text-deep">{site.email}</a>
              </Detail>
              <Detail icon={Phone} title="Phone">
                <a href={`tel:${site.phone.replace(/\s/g, "")}`} className="hover:text-deep">{site.phone}</a>
              </Detail>
              <Detail icon={MapPin} title="Address" wide>
                {site.address}
              </Detail>
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
              <label htmlFor="c-phone" className={label}>Phone <span className="normal-case tracking-normal">(optional)</span></label>
              <input id="c-phone" name="phone" type="tel" autoComplete="tel" className={field} />
            </div>
            <div className="sm:col-span-2">
              <label htmlFor="c-message" className={label}>Message</label>
              <textarea id="c-message" name="message" rows={5} required className={`${field} resize-y`} />
            </div>
            <button
              type="submit"
              className="rounded-xl bg-slate-900 px-6 py-4 font-mono text-xs tracking-[0.2em] text-white uppercase transition hover:bg-deep focus-visible:outline-deep active:scale-[0.99] sm:col-span-2"
            >
              Submit
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
