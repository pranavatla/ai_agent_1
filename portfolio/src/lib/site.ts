// Every piece of copy and every image on the site lives here, so it can be replaced in one edit.
// Claims are kept to what the résumé and the live projects can back up: no invented metrics.
import type { IconType } from "react-icons";
import { Bot, Cloud, DatabaseZap, Siren, Users, Workflow, type LucideIcon } from "lucide-react";
import { FaGithub, FaLinkedinIn } from "react-icons/fa6";
import { MdAlternateEmail } from "react-icons/md";

export const site = {
  brand: "Atla",
  name: "Sai Pranav Atla",
  tagline: "10+ years keeping enterprise cloud running. Now building the AI automation that helps run it.",
  handle: "Cloud & DevOps · Operations · Applied AI · Open to roles",
  description:
    "Sai Pranav Atla: cloud infrastructure specialist with 10+ years across IBM, TCS and Accenture. SAP on AWS, platform operations, incident leadership and AI automation. Based in Bengaluru, open to cloud, DevOps and AI engineering roles.",
  email: "pranavatla@gmail.com",
  phone: "+91 96322 98045",
  address: "Bengaluru, India",
  url: "https://atla.in",
  linkedin: "https://linkedin.com/in/saipranavatla/",
  // Transparent cut-out (WebP with alpha), 896x1195; About's frame uses this aspect ratio.
  portrait: "/media/portrait.webp",
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

// Each phrase pairs with a tile naming the real tools behind it (from the résumé).
export const services: { phrase: string; color: string; icon: LucideIcon; tools: string }[] = [
  { phrase: "Run SAP on AWS.", color: "#1d6fd0", icon: Cloud, tools: "S/4HANA · EC2 · VPC · Route 53" },
  { phrase: "Automate Infra.", color: "#6d3bd4", icon: Workflow, tools: "Terraform · Ansible · Jenkins · GitHub Actions" },
  { phrase: "Tame Incidents.", color: "#0f7c8a", icon: Siren, tools: "Major incidents · RCA · ITSM" },
  { phrase: "Build RAG Apps.", color: "#a86a12", icon: DatabaseZap, tools: "Bedrock · ChromaDB · Embeddings" },
  { phrase: "Ship AI Agents.", color: "#c2306b", icon: Bot, tools: "Multi-agent workflows · MCP" },
  { phrase: "Lead Teams.", color: "#157a5a", icon: Users, tools: "25+ engineers · SLA/KPI governance" },
];

export const about = {
  statement: ["systems", "that", "hold."],
  bio: "For 10+ years I’ve kept enterprise services running, from severity-1 bridges at IBM to SAP’s own Customer Experience platforms at Accenture, where I lead a 16-member operations team, work across cross-functional groups of 25+ engineers and hold 100% SLA compliance. Outside work I build on my own AWS account and ship every project end to end.",
};

export const experience = [
  {
    period: "2021 — Now",
    role: "Cloud Infrastructure Specialist, Platform & AI Automation",
    org: "Accenture · client: SAP",
    blurb: "Single point of contact between SAP and Accenture for the SAP CX and I&CX portfolio. Lead a 16-member operations team at 100% SLA compliance, ran the zero-downtime transition from NTT Data, and built the automated SLA/KPI signalling platform behind the SAP Hero Award and a place in Accenture’s Top 25 Global AI Programs.",
    stack: "AWS · SAP S/4HANA · Commerce Cloud · C4C · Terraform · Jenkins",
  },
  {
    period: "2018 — 2021",
    role: "Subject Matter Expert, Cloud Operations",
    org: "Tata Consultancy Services · client: SAP",
    blurb: "Owned incident response for the SAP Customer Experience estate: cut resolution time 40% with triage automation and standard escalation paths, rebuilt the knowledge base to cut repeat issues 30%, and lifted operational efficiency 25%. Nine SAP Best Performer of the Month awards.",
    stack: "ITSM · Confluence · GitHub · Capacity planning",
  },
  {
    period: "2015 — 2018",
    role: "Major Incident Manager",
    org: "IBM",
    blurb: "Incident commander for severity-1 events on enterprise platforms: cross-team resolution, real-time executive communication and post-incident review. Brought agile practice into incident and problem management, reducing mean time to resolution.",
    stack: "Incident & problem management · ITSM",
  },
  {
    period: "Ongoing · Independent",
    role: "AI & cloud projects",
    org: "atla.in",
    blurb: "Gita Reflection: RAG over 700 verses on Amazon Bedrock, with a leakage test proving answers come from the retrieved source. An operations console with incident simulation. This site: Next.js on S3 and CloudFront, deployed by GitHub Actions with OIDC. Also publish free AWS certification study material.",
    stack: "Bedrock · Python · FastAPI · Next.js · GitHub Actions",
  },
];

export const timeline = { subtitle: "10+ years in cloud and service operations" };

// Everything here is on every current résumé.
export const recognition = {
  awards: [
    { name: "SAP Hero Award, Innovator", by: "SAP" },
    { name: "Top 25 Global AI Programs", by: "Accenture" },
    { name: "ACE, Accenture Celebrates Excellence", by: "Accenture" },
    { name: "9× Best Performer of the Month", by: "SAP" },
    { name: "Delivery Excellence Award 2019", by: "TCS" },
  ],
  certifications: [
    { name: "AWS Certified AI Practitioner", code: "AIF-C01" },
    { name: "HashiCorp Certified: Terraform Associate", code: "" },
    { name: "AWS Certified Cloud Practitioner", code: "CLF-C02" },
  ],
  education: "B.Tech, Computer Science & Engineering · CMR Institute of Technology, Bangalore · 2015",
};

// One résumé per role family; the file names say which is which.
export const resumes = [
  { label: "Cloud & DevOps", href: "/resume/Sai_Pranav_Atla_Resume_DevOps_Cloud.pdf" },
  { label: "Operations & Program Management", href: "/resume/Sai_Pranav_Atla_Resume_Operations_Program_Manager.pdf" },
  { label: "Service Delivery (SAP)", href: "/resume/Sai_Pranav_Atla_Resume_Service_Delivery.pdf" },
  { label: "Operations Leadership", href: "/resume/Sai_Pranav_Atla_Resume_Operations_Leader.pdf" },
];

export const contact = {
  lead: "Open to Cloud & DevOps, Operations Management and Applied AI roles in Bengaluru, and always up for a good conversation about running reliable systems. The fastest way to reach me is email.",
};

export const capabilities = ["Cloud & platform operations", "SAP on AWS", "AI automation & RAG", "Incident & service leadership"];

export const sections = [
  { id: "home", label: "Home" },
  { id: "work", label: "Work" },
  { id: "services", label: "Services" },
  { id: "about", label: "About" },
  { id: "contact", label: "Contact" },
] as const;
