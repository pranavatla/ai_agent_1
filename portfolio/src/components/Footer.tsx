import { EnvelopeSimple, FileArrowDown, MapPin, Phone } from "@phosphor-icons/react/dist/ssr";
import { capabilities, sections, site, socials } from "@/lib/site";

const heading = "text-sm font-semibold text-slate-900";

export default function Footer() {
  return (
    <footer id="footer" data-covers-galaxy className="relative overflow-hidden px-6 pt-24 sm:px-10 md:pr-24">
      <div aria-hidden className="galaxy pointer-events-none absolute -bottom-[40vw] left-1/2 aspect-[2/1] w-[130vw] -translate-x-1/2 opacity-70 blur-[70px]" />
      <div className="relative z-10 mx-auto grid max-w-6xl gap-12 sm:grid-cols-2 lg:grid-cols-[1.5fr_1fr_1fr_1.3fr]">
        <div>
          <p className="font-display text-2xl font-bold tracking-[0.08em] uppercase">{site.brand}</p>
          <p className="mt-4 max-w-xs text-sm leading-relaxed text-slate-600">{site.tagline}</p>
        </div>
        <nav aria-label="Footer">
          <p className={heading}>Explore</p>
          <ul className="mt-5 space-y-3 text-sm">
            {sections.map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`} className="hit text-slate-700 transition hover:text-deep">{s.label}</a>
              </li>
            ))}
          </ul>
        </nav>
        <div>
          <p className={heading}>What I do</p>
          <ul className="mt-5 space-y-3 text-sm text-slate-700">
            {capabilities.map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
        <div>
          <p className={heading}>Contact</p>
          <ul className="mt-5 space-y-3 text-sm text-slate-700">
            <li className="flex items-center gap-3">
              <EnvelopeSimple aria-hidden className="size-4 shrink-0 text-deep" />
              <a href={`mailto:${site.email}`} className="hit break-all transition hover:text-deep">{site.email}</a>
            </li>
            <li className="flex items-center gap-3">
              <Phone aria-hidden className="size-4 shrink-0 text-deep" />
              <a href={`tel:${site.phone.replace(/\s/g, "")}`} className="hit transition hover:text-deep">{site.phone}</a>
            </li>
            <li className="flex items-center gap-3">
              <MapPin aria-hidden className="size-4 shrink-0 text-deep" />
              {site.address}
            </li>
            <li className="flex items-center gap-3">
              <FileArrowDown aria-hidden className="size-4 shrink-0 text-deep" />
              <a href="#resume" className="hit transition hover:text-deep">Résumés</a>
            </li>
          </ul>
        </div>
      </div>

      {/* Final band: the outline wordmark sits behind the socials and copyright, cropped at the baseline. */}
      <div className="relative flow-root -mx-6 mt-20 sm:-mx-10 md:-mr-24">
        <p
          aria-hidden
          className="font-display relative -mb-[0.1em] text-center leading-[0.74] font-bold tracking-[-0.02em] text-transparent uppercase select-none [-webkit-text-stroke:1px_rgba(29,111,208,0.25)]"
          style={{ fontSize: "39.3vw" }}
        >
          {site.brand}
        </p>
        <div className="absolute inset-x-0 bottom-6 z-10 mx-auto flex max-w-6xl px-6 sm:px-10 items-end justify-between gap-4">
          <ul className="flex gap-2">
            {socials.filter((s) => s.label !== "Email").map(({ label, href, icon: Icon }) => (
              <li key={label}>
                <a
                  href={href}
                  target={href.startsWith("http") ? "_blank" : undefined}
                  rel="noopener noreferrer"
                  aria-label={label}
                  className="grid size-11 place-items-center rounded-xl border border-slate-200 bg-surface/70 text-slate-700 shadow-sm backdrop-blur-md transition duration-200 ease-snappy hover:border-deep/50 hover:text-deep active:scale-[0.96]"
                >
                  <Icon aria-hidden className="size-4" />
                </a>
              </li>
            ))}
          </ul>
          <p className="text-right font-mono text-xs text-slate-500">
            © {new Date().getFullYear()} {site.name}
          </p>
        </div>
      </div>

    </footer>
  );
}
