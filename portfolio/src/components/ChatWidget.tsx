"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { ChatCircleDots, PaperPlaneRight, X } from "@phosphor-icons/react/dist/ssr";
import { site } from "@/lib/site";

type Turn = { role: "user" | "assistant"; content: string };

const STARTERS = ["What has Pranav built?", "What did he do at Accenture?", "Which certifications does he hold?"];
const MAX_CHARS = 500;
const HISTORY = 6;

// Answers are shown as plain text (React escapes it), so strip the model's bold markers.
const tidy = (text: string) => text.replace(/\*\*/g, "");

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) input.current?.focus();
  }, [open]);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" });
  }, [turns, busy, error]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    const next: Turn[] = [...turns, { role: "user", content: text }];
    setTurns(next);
    setDraft("");
    setError("");
    setBusy(true);
    try {
      const res = await fetch(site.chatUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: next.slice(-HISTORY) }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && typeof data.reply === "string") {
        setTurns([...next, { role: "assistant", content: tidy(data.reply) }]);
      } else {
        // Roll the question back so the conversation stays user/assistant alternating.
        setTurns(turns);
        setDraft(text);
        setError(data.error ?? "The assistant is busy right now. Please try again in a minute.");
      }
    } catch {
      setTurns(turns);
      setDraft(text);
      setError("Couldn't reach the assistant. Check your connection and try again.");
    } finally {
      setBusy(false);
      input.current?.focus();
    }
  }

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    ask(draft);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="atla-chat"
        aria-label={open ? "Close the assistant" : "Ask the assistant about Pranav"}
        className="fixed right-5 bottom-5 z-50 grid size-14 place-items-center rounded-full bg-deep text-white shadow-[0_12px_30px_rgba(15,23,42,0.25)] transition hover:scale-105 active:scale-95"
      >
        {open ? <X aria-hidden className="size-6" /> : <ChatCircleDots aria-hidden className="size-7" />}
      </button>

      {open && (
        <section
          id="atla-chat"
          role="dialog"
          aria-label="Ask about Pranav"
          className="fixed right-5 bottom-24 z-50 flex max-h-[min(34rem,calc(100dvh-8rem))] w-[min(24rem,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-surface text-slate-900 shadow-[0_24px_60px_rgba(15,23,42,0.18)]"
        >
          <header className="border-b border-slate-200 px-5 py-4">
            <p className="font-semibold">Ask about Pranav</p>
            <p className="mt-0.5 text-xs text-slate-500">{"AI answers grounded in his profile, served through gate.atla.in."}</p>
          </header>

          <div ref={log} aria-live="polite" className="flex-1 space-y-3 overflow-y-auto px-5 py-4 text-[15px] leading-relaxed">
            {turns.length === 0 && (
              <div className="space-y-2">
                <p className="text-slate-600">{"Hi! Ask me about Pranav's work, projects or experience."}</p>
                {STARTERS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => ask(q)}
                    className="block w-full rounded-xl border border-slate-200 px-3 py-2 text-left text-sm transition hover:border-deep hover:text-deep"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            {turns.map((t, i) => (
              <p
                key={i}
                className={`rounded-2xl px-4 py-2.5 whitespace-pre-wrap ${t.role === "user" ? "ml-8 bg-deep text-white" : "mr-8 bg-slate-100"}`}
              >
                {t.content}
              </p>
            ))}
            {busy && <p className="mr-8 rounded-2xl bg-slate-100 px-4 py-2.5 text-slate-500">{"Thinking…"}</p>}
            {error && (
              <p role="alert" className="text-sm text-red-600">
                {error}
              </p>
            )}
          </div>

          <form onSubmit={onSubmit} className="flex items-center gap-2 border-t border-slate-200 px-3 py-3">
            <input
              ref={input}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={MAX_CHARS}
              placeholder="Type a question…"
              aria-label="Your question"
              className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm focus:border-deep focus:ring-2 focus:ring-deep/20 focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy || !draft.trim()}
              aria-label="Send"
              className="grid size-10 place-items-center rounded-xl bg-deep text-white transition disabled:opacity-40"
            >
              <PaperPlaneRight aria-hidden className="size-5" />
            </button>
          </form>
          <p className="px-5 pb-3 text-[11px] text-slate-500">
            {"Please don't share personal details. For anything else, "}
            <a href={site.linkedin} target="_blank" rel="noopener noreferrer" className="underline hover:text-deep">
              message Pranav on LinkedIn
            </a>
            .
          </p>
        </section>
      )}
    </>
  );
}
