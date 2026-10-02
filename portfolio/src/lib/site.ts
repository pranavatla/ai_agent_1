// Every piece of copy and every image on the site lives here, so it can be replaced in one edit.
// Claims are kept to what the résumé and the live projects can back up: no invented metrics.
// One icon family site-wide: Phosphor (the SSR entry works in server and client components).
import type { Icon } from "@phosphor-icons/react";
import { Cloud, Database, EnvelopeSimple, FlowArrow, GithubLogo, LinkedinLogo, Robot, Siren, UsersThree } from "@phosphor-icons/react/dist/ssr";

export const site = {
  brand: "Atla",
  name: "Sai Pranav Atla",
  tagline: "10+ years in enterprise operations. Bringing that experience to cloud automation and applied AI. Based in Bengaluru.",
  // Rendered as separate items with spacing, not joined by separators.
  handle: ["Cloud operations", "Service delivery", "Applied AI", "Open to opportunities"],
  description:
    "Sai Pranav Atla’s portfolio: 10+ years in enterprise operations across IBM, TCS and Accenture, with experience in SAP CX/I&CX, service delivery and independent cloud and AI projects. Based in Bengaluru.",
  email: "pranavatla@gmail.com",
  phone: "+91 96322 98045",
  address: "Bengaluru, India",
  url: "https://atla.in",
  // Public Lambda Function URL for the chat widget (no secret: the gateway key stays in the Lambda).
  chatUrl: "https://vi4tzxl3vftteec2xphilvtxqy0kbzbd.lambda-url.ap-south-1.on.aws/",
  linkedin: "https://linkedin.com/in/saipranavatla/",
  // Transparent cut-out (WebP with alpha), 896x1195; About's frame uses this aspect ratio.
  portrait: "/media/portrait.webp",
};

export const socials: { label: string; href: string; icon: Icon }[] = [
  { label: "GitHub", href: "https://github.com/pranavatla", icon: GithubLogo },
  { label: "LinkedIn", href: site.linkedin, icon: LinkedinLogo },
  { label: "Email", href: `mailto:${site.email}`, icon: EnvelopeSimple },
];

export const work = {
  subtitle: "Independent projects · built with AI assistance, hosted on AWS",
};

export const projects: { title: string; tag: string; href: string; image: string; video?: string; alt: string }[] = [
  {
    title: "Gita Reflection",
    tag: "RAG · Amazon Bedrock",
    href: "https://gita.atla.in/",
    image: "/media/gita-brag.jpg",
    video: "/media/gita-brag.mp4",
    alt: "Gita Reflection answering a question with the source Bhagavad Gita verse alongside",
  },
  {
    title: "Gate · LLM Gateway",
    tag: "Multi-provider AI · Budgets, policy & failover",
    href: "https://gate.atla.in/",
    image: "/media/gate-brag.jpg",
    video: "/media/gate-brag.mp4",
    alt: "Gate, one API for Anthropic, OpenAI, Gemini and Amazon Bedrock with tenant controls, budgets and audit logging",
  },
  {
    title: "Browser Arcade",
    tag: "JavaScript Canvas · Built with Claude",
    href: "https://games.atla.in/",
    image: "/media/games-brag.jpg",
    video: "/media/games-brag.mp4",
    alt: "Drop, Invaders and Break, three browser games",
  },
  {
    title: "AIF-C01 Study Guide",
    tag: "AWS AI Practitioner · Study guide",
    href: "https://aif.atla.in/",
    image: "/media/aif-brag.jpg",
    video: "/media/aif-brag.mp4",
    alt: "Four-day AWS AI Practitioner (AIF-C01) study guide with practice-exam questions",
  },
  {
    title: "Observability Deck",
    tag: "Live observability · Tokens, traffic & uptime",
    href: "https://obs.atla.in/",
    image: "/media/obs.jpg",
    video: "/media/obs-brag.mp4",
    alt: "Observability deck showing live site health, traffic, latency, errors, security and AI usage",
  },
];

// Each phrase pairs with a tile naming the real tools behind it (from the résumé).
// Each phrase has its own colour, shared with its tile on the right.
// One accent for the whole site: every tile shares --deep rather than its own hue.
export const services: { phrase: string; color: string; icon: Icon; tools: string }[] = [
  { phrase: "Run cloud ops.", color: "#1d6fd0", icon: Cloud, tools: "SAP CX / I&CX, AWS, Service health" },
  { phrase: "Automate infra.", color: "#1d6fd0", icon: FlowArrow, tools: "Terraform, Ansible, Jenkins, GitHub Actions" },
  { phrase: "Lead incidents.", color: "#1d6fd0", icon: Siren, tools: "Major incidents, RCA, ITIL 4" },
  { phrase: "Build RAG apps.", color: "#1d6fd0", icon: Database, tools: "Bedrock, ChromaDB, Embeddings" },
  { phrase: "Build AI tools.", color: "#1d6fd0", icon: Robot, tools: "Python, FastAPI, AI workflows" },
  { phrase: "Lead teams.", color: "#1d6fd0", icon: UsersThree, tools: "16-member team, SLA/KPI governance" },
];

export const about = {
  statement: ["calm", "under", "load."],
  bio: "My work connects technology, teams and the people who rely on a service. Across IBM, TCS and Accenture, I’ve led critical incident response and supported SAP cloud operations. At Accenture, I led a 16-member team across SAP CX and I&CX until April 2026. Outside work, I use AI as a building partner to explore cloud automation, retrieval and observability on my own AWS infrastructure.",
};

export const experience = [
  {
    period: "Jan 2021 - Apr 2026",
    role: "Cloud & Platform Infrastructure Specialist",
    org: "Accenture · client: SAP",
    blurb: "Led a 16-member operations team and served as the point of contact between SAP and Accenture for the CX and I&CX portfolio. My work spanned SLA/KPI governance, capacity planning and scope assessment, with 100% SLA compliance across the supported services. Delivered automated operational reporting recognised with the SAP Hero Award and selection into Accenture’s Top 25 Global AI Programs.",
    stack: "SAP CX / I&CX, Commerce Cloud, C4C, SLA/KPI governance",
  },
  {
    period: "Sep 2018 - Jan 2021",
    role: "Subject Matter Expert, Cloud Operations",
    org: "Tata Consultancy Services · client: SAP",
    blurb: "Managed incident response for SAP Customer Experience services. Triage automation and standard escalation paths reduced resolution time by 40%; a rebuilt knowledge base reduced repeat-issue resolution time by 30%. Process and tooling improvements increased operational efficiency by 25%. Received nine SAP Best Performer of the Month awards.",
    stack: "ITSM, Confluence, GitHub, Capacity planning",
  },
  {
    period: "Aug 2015 - Sep 2018",
    role: "Major Incident Manager",
    org: "IBM",
    blurb: "Coordinated the response to severity-1 incidents on enterprise platforms, bringing technical teams together under pressure and keeping customers and executives informed. Led post-incident reviews and introduced improvements to incident and problem management to reduce resolution time.",
    stack: "Incident & problem management, ITSM",
  },
  {
    period: "Ongoing · Independent",
    role: "AI & cloud projects",
    org: "atla.in",
    blurb: "Build and host projects on my own AWS infrastructure: Gita Reflection explores retrieval across 700 verses, the Observability Deck brings together live site health, traffic, latency, security and AI usage, and the Browser Arcade experiments with AI-assisted coding. I test retrieval grounding and automate deployments, including this Next.js portfolio on S3 and CloudFront.",
    stack: "Bedrock, Python, FastAPI, Next.js, GitHub Actions",
  },
];

export const timeline = { subtitle: "10+ years across incident, cloud and service operations" };

// Everything here is on every current résumé.
export const recognition = {
  awards: [
    { name: "SAP Hero Award (Innovator)", by: "SAP" },
    { name: "Top 25 Global AI Programs", by: "Accenture" },
    { name: "ACE (Accenture Celebrates Excellence)", by: "Accenture" },
    { name: "9× Best Performer of the Month", by: "SAP" },
    { name: "Delivery Excellence Award 2019", by: "TCS" },
  ],
  // Confirmed by Sai Pranav and his résumés. Years omitted until each issue date is confirmed.
  certifications: [
    { name: "AWS Certified AI Practitioner", meta: "AWS · AIF-C01" },
    { name: "AWS Certified Cloud Practitioner", meta: "AWS · CLF-C02" },
    { name: "Microsoft Certified: Azure Fundamentals", meta: "Microsoft · AZ-900" },
    { name: "HashiCorp Certified: Terraform Associate", meta: "HashiCorp" },
    { name: "ITIL 4 Foundation", meta: "AXELOS" },
  ],
  education: "B.Tech in Computer Science & Engineering, CMR Institute of Technology, Bengaluru (2015)",
};

// One résumé per role family; the file names say which is which.
export const resumes = [
  { label: "Cloud & DevOps", href: "/resume/Sai_Pranav_Atla_Resume_DevOps_Cloud.pdf" },
  { label: "Operations & Program Management", href: "/resume/Sai_Pranav_Atla_Resume_Operations_Program_Manager.pdf" },
  { label: "Service Delivery (SAP)", href: "/resume/Sai_Pranav_Atla_Resume_Service_Delivery.pdf" },
  { label: "Operations Leadership", href: "/resume/Sai_Pranav_Atla_Resume_Operations_Leader.pdf" },
];

export const contact = {
  lead: "Open to roles and collaborations in cloud operations, service delivery, DevOps and applied AI. Based in Bengaluru. Email is the easiest way to reach me.",
};

export const capabilities = ["Cloud & platform operations", "SAP CX & I&CX operations", "AI automation & RAG", "Incident & service leadership"];

export const sections = [
  { id: "home", label: "Home" },
  { id: "work", label: "Work" },
  { id: "services", label: "What I do" },
  { id: "about", label: "About" },
  { id: "contact", label: "Contact" },
] as const;
