# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Recruiters and hiring managers filling **Cloud & DevOps**, **Operations Management** and **Applied AI** roles, primarily in Bengaluru (Sai Pranav is not open to relocation). They arrive from LinkedIn, a résumé, or a direct link, skim in a minute or two, and decide whether to reach out.

Success: the visitor contacts Sai Pranav by email, phone or LinkedIn, or shortlists him for a conversation.

## Product Purpose

atla.in is Sai Pranav Atla's personal portfolio. It shows a recruiter, quickly and credibly, that he combines 10+ years of enterprise cloud and service operations with hands-on applied AI work he designs, builds and runs himself.

## Positioning

An operations leader who ships. Most operations managers don't build; most AI builders haven't run major incidents or SAP platforms at enterprise scale. Sai Pranav has done both: major-incident management at IBM, cloud operations at TCS, platform and AI automation at Accenture leading 25+ engineers, and live AI projects on his own AWS account.

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

Confirmed career facts:

| Period | Role | Organisation |
| --- | --- | --- |
| Aug 2015 – 2018 | Major Incident Manager | IBM |
| Sep 2018 – Jan 2021 | Subject Matter Expert, Cloud Operations | Tata Consultancy Services |
| Jan 2021 – now | Cloud Infrastructure Specialist, Platform & AI Automation | Accenture (leads cross-functional teams of 25+ engineers) |

- Skills (from résumé): AWS (EC2, S3, IAM, VPC, CloudFront, Route 53, SQS, DynamoDB), SAP S/4HANA on AWS, Terraform, Ansible, Jenkins, Docker, Kubernetes, Python, FastAPI, RAG (ChromaDB), agentic and multi-agent workflows with MCP, major incident and problem management, RCA, SLA/KPI governance, ITSM.
- Live projects: Gita Reflection (RAG over 700 verses on Amazon Bedrock), an Operations Console (FastAPI; telemetry is simulated, not production), and a Browser Arcade (built with Claude). Screens in `public/media/`.

**Pending: do not fabricate or placeholder these until supplied.**

- Résumé PDF (to be linked for download).
- Quantified outcomes Sai Pranav can defend (e.g. MTTR, cost, uptime).
- Certifications (names and years).
- Exact IBM end month (2018 confirmed, month not).

## Product Principles

1. **Credibility over flourish.** Every claim traces to the résumé or a live project.
2. **Operations first, AI as the edge.** Lead with a decade of running enterprise services; applied AI is the differentiator, not the whole story.
3. **Show, don't list.** Live, clickable projects outweigh skill badges.
4. **One minute to "yes".** Role, seniority, location and a way to make contact must be clear without scrolling far.

## Accessibility & Inclusion

WCAG 2.2 AA. Every scroll-driven effect has a static, fully readable fallback under `prefers-reduced-motion`.
