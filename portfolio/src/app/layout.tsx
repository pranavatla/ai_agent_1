import type { Metadata, Viewport } from "next";
import { Manrope, Space_Grotesk, Space_Mono } from "next/font/google";
import { MotionConfig } from "motion/react";
import { site } from "@/lib/site";
import "./globals.css";

// Variable names must match the tokens globals.css reads, or every font silently falls back.
const display = Manrope({ variable: "--font-manrope", subsets: ["latin"], weight: ["500", "700", "800"] });
const sans = Space_Grotesk({ variable: "--font-grotesk", subsets: ["latin"] });
const mono = Space_Mono({ variable: "--font-spacemono", subsets: ["latin"], weight: ["400", "700"] });

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: `${site.name} — Cloud, platform & AI automation`,
  description: site.description,
  alternates: { canonical: "/" },
  icons: { icon: "/media/atla-mark.svg" },
  openGraph: {
    type: "website",
    url: "/",
    title: `${site.name} — Cloud, platform & AI automation`,
    description: site.description,
    images: "/media/portfolio-social.png",
  },
  twitter: { card: "summary_large_image", images: "/media/portfolio-social.png" },
};

export const viewport: Viewport = { themeColor: "#f7f8fc" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${sans.variable} ${mono.variable} antialiased`}>
      <body>
        <MotionConfig reducedMotion="user">{children}</MotionConfig>
      </body>
    </html>
  );
}
