"use client";

import { useEffect } from "react";

// CloudWatch RUM: page-load timing, Core Web Vitals and browser errors from real visitors,
// read on the obs.atla.in Grafana dashboard. These IDs are public by design: the Cognito guest
// identity can only send RUM events for this app monitor, and only from the listed domains.
const APP_MONITOR_ID = "a80555a8-620f-466d-bdde-43a7b94e2ad6";
const REGION = "us-east-1";
const IDENTITY_POOL_ID = "us-east-1:9288d345-90d7-4e93-a12d-0d3c74a1469d";
const HOSTS = new Set(["atla.in", "www.atla.in"]);

export default function Rum() {
  useEffect(() => {
    // Local builds and previews would only add noise to production data.
    if (!HOSTS.has(window.location.hostname)) return;

    // Loaded after hydration so monitoring never delays the page it measures.
    import("aws-rum-web")
      .then(({ AwsRum }) => {
        new AwsRum(APP_MONITOR_ID, "1.0.0", REGION, {
          sessionSampleRate: 1,
          identityPoolId: IDENTITY_POOL_ID,
          endpoint: `https://dataplane.rum.${REGION}.amazonaws.com`,
          telemetries: ["performance", "errors", "http"],
          allowCookies: false,
          enableXRay: false,
          signing: true,
        });
      })
      .catch(() => {
        // Monitoring must never break the site; a blocked or failed load is simply not recorded.
      });
  }, []);

  return null;
}
