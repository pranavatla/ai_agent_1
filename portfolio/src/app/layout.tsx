import type { Metadata, Viewport } from "next";
import { Manrope, Space_Grotesk, Space_Mono } from "next/font/google";
import { MotionConfig } from "motion/react";
import { site } from "@/lib/site";
import { THEME_SCRIPT } from "@/lib/theme-script";
import Rum from "@/components/Rum";
import "./globals.css";

// Variable names must match the tokens globals.css reads, or every font silently falls back.
const display = Manrope({ variable: "--font-manrope", subsets: ["latin"], weight: ["500", "700", "800"] });
const sans = Space_Grotesk({ variable: "--font-grotesk", subsets: ["latin"] });
const mono = Space_Mono({ variable: "--font-spacemono", subsets: ["latin"], weight: ["400", "700"] });

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: `${site.name} - Cloud operations, service delivery & AI`,
  description: site.description,
  alternates: { canonical: "/" },
  icons: { icon: "/media/atla-mark.svg" },
  openGraph: {
    type: "website",
    url: "/",
    title: `${site.name} - Cloud operations, service delivery & AI`,
    description: site.description,
    images: "/media/portfolio-social.png",
  },
  twitter: { card: "summary_large_image", images: "/media/portfolio-social.png" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f8fc" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0f1a" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${display.variable} ${sans.variable} ${mono.variable} antialiased`}>
      <head>
        {/* Sets data-theme before first paint; the attribute is why <html> suppresses the hydration warning. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <MotionConfig reducedMotion="user">{children}</MotionConfig>
        <Rum />
      </body>
    </html>
  );
}
