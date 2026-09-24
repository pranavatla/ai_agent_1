export default function Background() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {/* Subtle top ambient neutral lighting */}
      <div
        className="absolute inset-x-0 top-0 h-[600px] opacity-60"
        style={{
          background: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(29, 111, 208, 0.08), transparent 70%)",
        }}
      />
      {/* Subtle architectural grid */}
      <div className="subtle-grid absolute inset-0 opacity-40 [mask-image:radial-gradient(ellipse_60%_50%_at_50%_30%,#000_70%,transparent_100%)]" />
    </div>
  );
}
