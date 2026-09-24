import { EnvelopeSimple, FileArrowDown, MapPin, Phone } from "@phosphor-icons/react/dist/ssr";
import { capabilities, sections, site, socials } from "@/lib/site";

const heading = "font-mono text-xs font-semibold tracking-wider text-slate-500 uppercase";

export default function Footer() {
  return (
    <footer id="footer" className="bg-white border-t border-slate-200 px-6 pt-16 pb-12 sm:px-10">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <a href="#home" className="font-display text-2xl font-bold tracking-tight text-slate-900 hover:text-deep transition">
              {site.brand}
              <span className="text-deep font-mono text-base ml-1">.in</span>
            </a>
            <p className="mt-3 text-sm leading-relaxed text-slate-600 max-w-xs">
              {site.tagline}
            </p>
          </div>

          <div>
            <p className={heading}>Navigation</p>
            <ul className="mt-4 space-y-2.5 text-sm">
              {sections.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className="text-slate-600 hover:text-deep transition">
                    {s.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className={heading}>Core Focus</p>
            <ul className="mt-4 space-y-2.5 text-sm text-slate-600">
              {capabilities.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </div>

          <div>
            <p className={heading}>Direct Contact</p>
            <ul className="mt-4 space-y-3 text-sm text-slate-600">
              <li className="flex items-center gap-2.5">
                <EnvelopeSimple size={16} className="text-deep shrink-0" />
                <a href={`mailto:${site.email}`} className="hover:text-deep transition">
                  {site.email}
                </a>
              </li>
              <li className="flex items-center gap-2.5">
                <Phone size={16} className="text-deep shrink-0" />
                <a href={`tel:${site.phone.replace(/\s/g, "")}`} className="hover:text-deep transition">
                  {site.phone}
                </a>
              </li>
              <li className="flex items-center gap-2.5">
                <MapPin size={16} className="text-deep shrink-0" />
                <span>{site.address}</span>
              </li>
              <li className="flex items-center gap-2.5">
                <FileArrowDown size={16} className="text-deep shrink-0" />
                <a href="#resume" className="hover:text-deep transition">
                  Targeted Résumés
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-14 pt-8 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="font-mono text-xs text-slate-500">
            © {new Date().getFullYear()} {site.name}. All rights reserved.
          </p>

          <div className="flex items-center gap-2">
            {socials.map(({ label, href, icon: Icon }) => (
              <a
                key={label}
                href={href}
                target={href.startsWith("http") ? "_blank" : undefined}
                rel="noopener noreferrer"
                aria-label={label}
                className="grid size-9 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 transition hover:border-deep/40 hover:text-deep active:scale-95"
              >
                <Icon size={16} />
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
