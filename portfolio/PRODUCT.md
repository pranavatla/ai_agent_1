# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Recruiters and hiring managers filling **Cloud & DevOps**, **Operations Management** and **Applied AI** roles, primarily in Bengaluru. They arrive from LinkedIn, a résumé, or a direct link, skim in a minute or two, and decide whether to reach out.

Success: the visitor contacts Sai Pranav by email, phone or LinkedIn, or shortlists him for a conversation.

## Product Purpose

atla.in is Sai Pranav Atla's personal portfolio. It shows a recruiter, quickly and credibly, that he combines 10+ years of enterprise cloud and service operations with hands-on applied AI work he designs, builds and runs himself.

## Positioning

An operations leader who ships. Most operations managers don't build; most AI builders haven't run major incidents or SAP platforms at enterprise scale. Sai Pranav has done both: major-incident management at IBM, cloud operations at TCS, platform and service operations for SAP itself at Accenture (leading a 16-member team across cross-functional groups of 25+ engineers), and live AI projects on his own AWS account.

## Operating Context

- Visitors usually compare several candidates in a short session, often on a laptop, sometimes on a phone.
- The site is the hub for LinkedIn (`linkedin.com/in/saipranavatla`) and GitHub (`github.com/pranavatla`).
- Live project subdomains: `gita.atla.in`, `obs.atla.in`, `games.atla.in`.

## Capabilities and Constraints

- Single-page static site: Next.js static export, served from S3 + CloudFront, deployed by GitHub Actions on push to `develop`.
- No backend. The contact form hands off to the visitor's mail client via `mailto:`.
- All copy and imagery lives in `src/lib/site.ts`.

## Brand Commitments

- Name: **Sai Pranav Atla**. Short mark and domain: **Atla** / atla.in.
- Voice: plain, confident and specific. Name real tools and real responsibilities. No hype.
- Honesty is binding: only claims Sai Pranav can defend in an interview. No invented metrics, clients, testimonials or certifications.
- Portrait: `public/media/portrait.webp`, a cut-out from his own photo.

## Evidence on Hand

Source of truth: the five tailored résumés in `~/Downloads` (Sep 2026). Four are published in `public/resume/`.

| Period | Role | Organisation |
| --- | --- | --- |
| Aug 2015 – Sep 2018 | Major Incident Manager | IBM |
| Sep 2018 – Jan 2021 | Subject Matter Expert, Cloud / Service Operations | Tata Consultancy Services (client: SAP) |
| Jan 2021 – now | Cloud Infrastructure Specialist, Platform & AI Automation | Accenture (client: SAP CX, then I&CX from 2023) |

**Open conflict:** Sai Pranav confirmed TCS started **Sep 2018**; every résumé PDF says **Nov 2019**. The site uses Sep 2018. The PDFs need re-exporting to match.

Confirmed outcomes (on every résumé):

- TCS: resolution time −40% (triage automation, standard escalation paths); repeat issues −30% (knowledge-base rebuild); operational efficiency +25%; resourcing optimised 20%.
- Accenture: 100% SLA compliance under SAP's KPI frameworks; zero-downtime transition from NTT Data to Accenture; 16-member operations team; single point of contact between SAP and Accenture.
- Automated SLA/KPI signalling platform replacing manual reporting.

Recognition: SAP Hero Award (Innovator); Accenture Top 25 Global AI Programs; ACE (Accenture Celebrates Excellence); 9× SAP Best Performer of the Month; TCS Delivery Excellence Award 2019.

Certifications: AWS Certified AI Practitioner (AIF-C01); HashiCorp Certified: Terraform Associate; AWS Certified Cloud Practitioner (CLF-C02). Years not given.

Education: B.Tech, Computer Science & Engineering, CMR Institute of Technology, Bangalore, 2015.

Skills: AWS (EC2, S3, IAM, VPC, CloudFront, Route 53), SAP S/4HANA on AWS, SAP Commerce Cloud, C4C, Terraform, Ansible, Jenkins, GitHub Actions (OIDC), Docker, Kubernetes (working knowledge), Python, FastAPI, RAG, multi-agent workflows and MCP connectors, ITSM, RCA, SLA/KPI governance.

Live projects: Gita Reflection (`gita.atla.in`, RAG on Amazon Bedrock with a leakage test), Operations Console (`obs.atla.in`, simulated telemetry), Browser Arcade (`games.atla.in`). Screens in `public/media/`.

Not live, do not link: AtlaOps (`ops.atla.in` does not respond); the real-time market-data platform and 25-page RAG guide have no public URL yet.

**Pending or unresolved: do not fabricate.**

- Certification years.
- Relocation: the Operations Leader résumé says "open to Mumbai"; an earlier profile said not willing to relocate. The site says Bengaluru only.

## Product Principles

1. **Credibility over flourish.** Every claim traces to the résumé or a live project.
2. **Operations first, AI as the edge.** Lead with a decade of running enterprise services; applied AI is the differentiator, not the whole story.
3. **Show, don't list.** Live, clickable projects outweigh skill badges.
4. **One minute to "yes".** Role, seniority, location and a way to make contact must be clear without scrolling far.

## Accessibility & Inclusion

WCAG 2.2 AA. Every scroll-driven effect has a static, fully readable fallback under `prefers-reduced-motion`.
