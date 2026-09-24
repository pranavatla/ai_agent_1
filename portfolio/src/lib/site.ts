// Every piece of copy and every image on the site lives here, so it can be replaced in one edit.
// Claims are kept to what the résumé and the live projects can back up: no invented metrics.
import type { IconType } from "react-icons";
import { FaGithub, FaLinkedinIn } from "react-icons/fa6";
import { MdAlternateEmail } from "react-icons/md";

export const site = {
  brand: "Atla",
  name: "Sai Pranav Atla",
  tagline: "10+ years keeping enterprise cloud running. Now building the AI automation that helps run it.",
  handle: "Cloud, Platform & AIOps · Open to roles",
  description:
    "Sai Pranav Atla: cloud infrastructure specialist with 10+ years across IBM, TCS and Accenture. SAP on AWS, platform operations, incident leadership and AI automation. Based in Bengaluru, open to cloud, DevOps and AI engineering roles.",
  email: "pranavatla@gmail.com",
  phone: "+91 96322 98045",
  address: "Bengaluru, India",
  url: "https://atla.in",
  linkedin: "https://linkedin.com/in/saipranavatla/",
  // Transparent cut-out, 896x1195; About's frame uses this aspect ratio.
  portrait: "/media/portrait.png",
};

export const socials: { label: string; href: string; icon: IconType }[] = [
  { label: "GitHub", href: "https://github.com/pranavatla", icon: FaGithub },
  { label: "LinkedIn", href: site.linkedin, icon: FaLinkedinIn },
  { label: "Email", href: `mailto:${site.email}`, icon: MdAlternateEmail },
];

export const work = {
  subtitle: "Live projects · designed, built and run on my own AWS",
};

export const projects = [
  {
    title: "Gita Reflection",
    tag: "RAG · Amazon Bedrock",
    href: "https://gita.atla.in/",
    image: "/media/gita-poster.jpg",
    alt: "Gita Reflection answering a question with the source Bhagavad Gita verse alongside",
  },
  {
    title: "Operations Console",
    tag: "Observability · FastAPI",
    href: "https://obs.atla.in/",
    image: "/media/obs.jpg",
    alt: "Operations console with service health, latency, error rates and incident timeline",
  },
  {
    title: "Browser Arcade",
    tag: "JS Canvas · Built with Claude",
    href: "https://games.atla.in/",
    image: "/media/games.jpg",
    alt: "Drop, Invaders and Break, three browser games",
  },
];

export const services = [
  { phrase: "Run SAP on AWS.", color: "#1d6fd0", image: "https://picsum.photos/seed/service-1/600/600" },
  { phrase: "Automate Infra.", color: "#6d3bd4", image: "https://picsum.photos/seed/service-2/600/600" },
  { phrase: "Tame Incidents.", color: "#0f7c8a", image: "https://picsum.photos/seed/service-3/600/600" },
  { phrase: "Build RAG Apps.", color: "#a86a12", image: "https://picsum.photos/seed/service-4/600/600" },
  { phrase: "Ship AI Agents.", color: "#c2306b", image: "https://picsum.photos/seed/service-5/600/600" },
  { phrase: "Lead Teams.", color: "#157a5a", image: "https://picsum.photos/seed/service-6/600/600" },
];

export const about = {
  statement: ["systems", "that", "hold."],
  bio: "For 10+ years I’ve kept enterprise services running, from major-incident bridges at IBM to SAP cloud platforms at Accenture, where I work on infrastructure and AI-driven automation and lead cross-functional teams of 25+ engineers. Outside work I build on my own AWS account and ship every project end to end.",
};

export const experience = [
  {
    period: "2021 — Now",
    role: "Cloud Infrastructure Specialist, Platform & AI Automation",
    org: "Accenture",
    blurb: "Platform ownership for SAP CX and I&CX in the cloud: architecture standards, cost and capacity optimisation, SLA/KPI governance and AI-driven incident analysis. Reporting to enterprise and C-level stakeholders.",
    stack: "AWS · SAP S/4HANA · Terraform · Ansible · Jenkins · Kubernetes",
  },
  {
    period: "2019 — 2020",
    role: "Subject Matter Expert, Cloud Operations",
    org: "Tata Consultancy Services",
    blurb: "Go-to expert for cloud operations on enterprise accounts: major incident and problem management, root-cause analysis, and transition and transformation delivery.",
    stack: "ITSM · RCA · Cloud operations",
  },
  {
    period: "2015 — 2019",
    role: "Major Incident Manager",
    org: "IBM",
    blurb: "Ran major-incident bridges end to end: restoring service fast, coordinating resolver teams, leading RCA and keeping stakeholders informed under SLA pressure.",
    stack: "Incident & problem management · ITSM",
  },
  {
    period: "Ongoing · Independent",
    role: "AI & cloud projects",
    org: "atla.in",
    blurb: "Gita Reflection: RAG over 700 verses on Amazon Bedrock with reranking and source validation. An operations console with incident simulation. This site: Next.js on S3 and CloudFront, deployed by GitHub Actions with OIDC.",
    stack: "Bedrock · Python · FastAPI · Next.js · GitHub Actions",
  },
];

export const timeline = { subtitle: "10+ years in cloud and service operations" };

export const contact = {
  lead: "Open to cloud, DevOps, platform and AI engineering roles, and always up for a good conversation about running reliable systems. The fastest way to reach me is email.",
};

export const capabilities = ["Cloud & platform operations", "SAP on AWS", "AI automation & RAG", "Incident & service leadership"];

export const sections = [
  { id: "home", label: "Home" },
  { id: "work", label: "Work" },
  { id: "services", label: "Services" },
  { id: "about", label: "About" },
  { id: "contact", label: "Contact" },
] as const;
