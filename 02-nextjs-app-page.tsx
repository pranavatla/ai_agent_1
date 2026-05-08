/**
 * atla.in - Multi-Agent Portfolio & Showcase
 * Home page with agent overview and quick access
 */

'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Sparkles, Brain, Code, Target, BarChart3, Zap } from 'lucide-react';

interface Agent {
  agent_id: string;
  name: string;
  role: string;
  capabilities: string[];
  description: string;
}

export default function Home() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await fetch('http://localhost:8001/agents');
        const data = await response.json();
        setAgents(data.agents);
      } catch (err) {
        setError('Failed to load agents');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  const agentIcons: Record<string, React.ReactNode> = {
    portfolio: <Target className="w-8 h-8" />,
    technical: <Code className="w-8 h-8" />,
    strategic: <Brain className="w-8 h-8" />,
    analytics: <BarChart3 className="w-8 h-8" />,
    coordinator: <Zap className="w-8 h-8" />
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Hero Section */}
      <header className="relative overflow-hidden px-4 py-20 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 bg-slate-700/50 rounded-full border border-slate-600">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-slate-200">Multi-Agent AI System</span>
          </div>

          <h1 className="text-5xl sm:text-6xl font-bold mb-6 bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 bg-clip-text text-transparent">
            atla.in
          </h1>

          <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
            A production-grade multi-agent system showcasing specialized AI agents working together.
            Meet your portfolio, technical expertise, strategic thinking, and analytics capabilities.
          </p>

          <div className="flex flex-wrap gap-4 justify-center mb-12">
            <Link
              href="/playground"
              className="px-6 py-3 bg-amber-500 hover:bg-amber-600 text-white font-semibold rounded-lg transition-colors"
            >
              Try the Playground
            </Link>
            <Link
              href="/agents"
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors border border-slate-600"
            >
              Meet the Agents
            </Link>
          </div>

          <div className="text-sm text-slate-400">
            Built with FastAPI • Next.js • Multi-Agent Architecture
          </div>
        </div>
      </header>

      {/* Agents Grid */}
      <section className="px-4 py-16 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-12">
            <h2 className="text-3xl font-bold mb-4">Five Specialized Agents</h2>
            <p className="text-slate-400">
              Each agent specializes in a different domain, working together to showcase skills and capabilities.
            </p>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <p className="text-slate-400">Loading agents...</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-red-400">{error}</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {agents.map((agent) => (
                <Link
                  key={agent.agent_id}
                  href={`/agents/${agent.agent_id}`}
                  className="group p-6 rounded-lg bg-slate-700/50 border border-slate-600 hover:border-amber-500 hover:bg-slate-700 transition-all"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-slate-600 group-hover:bg-amber-500/20 transition-colors">
                        {agentIcons[agent.agent_id]}
                      </div>
                      <div>
                        <h3 className="font-bold text-lg">{agent.name}</h3>
                        <p className="text-sm text-slate-400 capitalize">{agent.role}</p>
                      </div>
                    </div>
                  </div>

                  <p className="text-slate-300 text-sm mb-4">{agent.description}</p>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {agent.capabilities.slice(0, 3).map((cap) => (
                      <span
                        key={cap}
                        className="text-xs px-2 py-1 bg-slate-600 rounded text-slate-200"
                      >
                        {cap.replace(/_/g, ' ')}
                      </span>
                    ))}
                    {agent.capabilities.length > 3 && (
                      <span className="text-xs px-2 py-1 bg-slate-600 rounded text-slate-200">
                        +{agent.capabilities.length - 3} more
                      </span>
                    )}
                  </div>

                  <button className="w-full py-2 px-4 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded font-medium transition-colors text-sm">
                    Chat with {agent.name.split(' ')[0]} →
                  </button>
                </Link>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Features */}
      <section className="px-4 py-16 sm:px-6 lg:px-8 bg-slate-800/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-12">Key Features</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: 'Specialized Agents',
                description:
                  'Five distinct agents (Portfolio, Technical, Strategic, Analytics, Coordinator) each with specialized knowledge and capabilities.'
              },
              {
                title: 'Agent Communication',
                description:
                  'Agents communicate via message bus, enabling complex multi-agent orchestration and collaboration.'
              },
              {
                title: 'Memory & Context',
                description:
                  'Each agent maintains conversation history and learns from interactions.'
              },
              {
                title: 'Live Integration',
                description:
                  'Connected to obs.atla.in for real-time operational data and analytics queries.'
              },
              {
                title: 'Interactive Playground',
                description:
                  'Try multi-agent scenarios and watch agents collaborate on complex queries.'
              },
              {
                title: 'Transparent Reasoning',
                description:
                  'See agent reasoning chains and decision-making process.'
              }
            ].map((feature, idx) => (
              <div
                key={idx}
                className="p-6 rounded-lg bg-slate-700/50 border border-slate-600 hover:border-amber-500/50 transition-colors"
              >
                <h3 className="font-bold text-lg mb-2">{feature.title}</h3>
                <p className="text-slate-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 py-16 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Explore?</h2>
          <p className="text-slate-300 mb-8">
            Try chatting with individual agents or use the playground to see them collaborate on complex tasks.
          </p>

          <div className="flex flex-wrap gap-4 justify-center">
            <Link
              href="/playground"
              className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-bold rounded-lg transition-all transform hover:scale-105"
            >
              Go to Playground
            </Link>
            <Link
              href="/agents"
              className="px-8 py-4 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors border border-slate-600"
            >
              Browse Agents
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-700 px-4 py-8 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto text-center text-slate-400 text-sm">
          <p>
            atla.in × obs.atla.in | Multi-Agent AI System | Built with FastAPI, Next.js, and TypeScript
          </p>
          <p className="mt-2">
            14-Week AI Journey | Pranav Atla
          </p>
        </div>
      </footer>
    </div>
  );
}
