// Every piece of copy and every image on the site lives here, so it can be replaced in one edit.
// Claims are kept to what the résumé and the live projects can back up: no invented metrics.
// One icon family site-wide: Phosphor (the SSR entry works in server and client components).
import type { Icon } from "@phosphor-icons/react";
import { Cloud, Database, EnvelopeSimple, FlowArrow, GithubLogo, LinkedinLogo, Robot, Siren, UsersThree } from "@phosphor-icons/react/dist/ssr";

export const site = {
  brand: "Atla",
  name: "Sai Pranav Atla",
  tagline: "10+ years in enterprise operations. Bringing that experience to cloud automation and applied AI.",
  // Rendered as separate items with spacing, not joined by separators.
  handle: ["Cloud operations", "Service delivery", "Applied AI", "Open to opportunities"],
  description:
    "Sai Pranav Atla’s portfolio: 10+ years in enterprise operations across IBM, TCS and Accenture, with experience in SAP CX/I&CX, service delivery and independent cloud and AI projects. Based in Bengaluru.",
  email: "pranavatla@gmail.com",
  phone: "+91 96322 98045",
  address: "Bengaluru, India",
  url: "https://atla.in",
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
    title: "Operations Console",
    tag: "Observability demo · FastAPI",
    href: "https://obs.atla.in/",
    image: "/media/obs.jpg",
    alt: "Operations console with service health, latency, error rates and incident timeline",
  },
  {
    title: "Browser Arcade",
    tag: "JavaScript Canvas · Built with Claude",
    href: "https://games.atla.in/",
    image: "/media/games-brag.jpg",
    video: "/media/games-brag.mp4",
    alt: "Drop, Invaders and Break, three browser games",
  },
];

// Each phrase pairs with a tile naming the real tools behind it (from the résumé).
export const services: { phrase: string; icon: Icon; tools: string }[] = [
  { phrase: "Run cloud ops.", icon: Cloud, tools: "SAP CX / I&CX, AWS, Service health" },
  { phrase: "Automate infra.", icon: FlowArrow, tools: "Terraform, Ansible, Jenkins, GitHub Actions" },
  { phrase: "Lead incidents.", icon: Siren, tools: "Major incidents, RCA, ITIL 4" },
  { phrase: "Build RAG apps.", icon: Database, tools: "Bedrock, ChromaDB, Embeddings" },
  { phrase: "Build AI tools.", icon: Robot, tools: "Python, FastAPI, AI workflows" },
  { phrase: "Lead teams.", icon: UsersThree, tools: "16-member team, SLA/KPI governance" },
];

export const about = {
  statement: ["systems", "that", "hold."],
  bio: "My work connects technology, teams and the people who rely on a service. Across IBM, TCS and Accenture, I’ve led critical incident response and supported SAP cloud operations. At Accenture, I lead a 16-member team across SAP CX and I&CX. Outside work, I use AI as a building partner to explore cloud automation, retrieval and observability on my own AWS infrastructure.",
};

export const experience = [
  {
    period: "Jan 2021 - Present",
    role: "Operations Lead: SAP CX & I&CX",
    org: "Accenture · client: SAP",
    blurb: "Lead a 16-member operations team and serve as the point of contact between SAP and Accenture for the CX and I&CX portfolio. My work spans SLA/KPI governance, capacity planning and scope assessment, with 100% SLA compliance across the supported services. Delivered automated operational reporting recognised with the SAP Hero Award and selection into Accenture’s Top 25 Global AI Programs.",
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
    blurb: "Build and host projects on my own AWS infrastructure: Gita Reflection explores retrieval across 700 verses, the Operations Console demonstrates incident simulation, and the Browser Arcade experiments with AI-assisted coding. I test retrieval grounding and automate deployments, including this Next.js portfolio on S3 and CloudFront.",
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
  lead: "I’m open to opportunities in cloud operations, service delivery, DevOps and applied AI, as well as project collaborations. Based in Bengaluru. If my experience fits what you’re working on, I’d welcome a conversation: email is the easiest way to reach me.",
};

export const capabilities = ["Cloud & platform operations", "SAP CX & I&CX operations", "AI automation & RAG", "Incident & service leadership"];

export const sections = [
  { id: "home", label: "Home" },
  { id: "work", label: "Work" },
  { id: "services", label: "Services" },
  { id: "about", label: "About" },
  { id: "contact", label: "Contact" },
] as const;
