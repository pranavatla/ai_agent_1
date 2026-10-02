import { EnvelopeSimple, GithubLogo, LinkedinLogo } from "@phosphor-icons/react/dist/ssr";
import { site, socials } from "@/lib/site";

const button =
  "inline-flex min-h-12 items-center justify-center gap-2 whitespace-nowrap rounded-full border border-slate-200 bg-surface px-6 py-3 text-sm font-medium text-slate-800 shadow-sm transition duration-200 ease-snappy hover:border-deep/40 hover:text-deep active:scale-[0.97]";
const primary =
  "inline-flex min-h-12 items-center justify-center gap-2 whitespace-nowrap rounded-full bg-deep px-6 py-3 text-sm font-semibold text-ground shadow-sm transition duration-200 ease-snappy hover:-translate-y-0.5 active:scale-[0.97]";

export default function Hero() {
  // Left-aligned in the page's content column; the right side stays open over the background band.
  return (
    <section id="home" className="relative flex min-h-[100dvh] items-center px-6 py-24 sm:px-10 md:pr-24">
      <div className="mx-auto w-full max-w-6xl">
        <ul className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs font-medium tracking-[0.2em] text-deep uppercase">
          {site.handle.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
        <h1 className="font-display mt-5 text-5xl font-extrabold tracking-tight text-slate-900 sm:text-7xl lg:text-8xl">
          {site.name}
        </h1>
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-slate-600 sm:text-lg">
          {site.tagline}
        </p>
        <div className="mt-8 grid grid-cols-2 gap-3 sm:flex sm:flex-wrap sm:items-center">
          <a href="#contact" className={`${primary} col-span-2 sm:col-span-1`}>
            <EnvelopeSimple aria-hidden weight="fill" className="size-4" />
            Get in touch
          </a>
          <a href={site.linkedin} target="_blank" rel="noopener noreferrer" className={button}>
            <LinkedinLogo aria-hidden weight="fill" className="size-4 text-deep" />
            LinkedIn
          </a>
          <a href={socials.find((social) => social.label === "GitHub")!.href} target="_blank" rel="noopener noreferrer" className={button}>
            <GithubLogo aria-hidden weight="fill" className="size-4 text-deep" />
            GitHub
          </a>
        </div>
      </div>
    </section>
  );
}
