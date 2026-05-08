"""
AtlaOps Multi-Agent Orchestrator Backend
FastAPI + Async Python for agent coordination and reasoning
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="AtlaOps Multi-Agent System",
    description="Orchestrates specialized agents for portfolio + technical showcase",
    version="1.0.0"
)

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class AgentRole(str, Enum):
    """Enum for agent specialized roles."""
    PORTFOLIO = "portfolio"
    TECHNICAL = "technical"
    STRATEGIC = "strategic"
    ANALYTICS = "analytics"
    COORDINATOR = "coordinator"


class Message(BaseModel):
    """Message format for agent communication."""
    sender_agent_id: str
    recipient_agent_id: Optional[str] = None  # None = broadcast
    message_type: str  # "request", "response", "context", "result"
    content: str
    context: Dict[str, Any] = {}
    timestamp: datetime = None
    conversation_id: str = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ChatRequest(BaseModel):
    """User query to be handled by agents."""
    user_id: str
    query: str
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None  # If None, coordinator routes


class ChatResponse(BaseModel):
    """Response from agent(s)."""
    conversation_id: str
    response: str
    agent_id: str
    agents_involved: List[str]
    reasoning_chain: Optional[List[str]] = None
    timestamp: datetime


# ============================================================================
# AGENT BASE CLASS
# ============================================================================

class Agent:
    """Base class for all specialized agents."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        capabilities: List[str],
        description: str = ""
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.description = description
        self.conversation_memory = []  # Recent conversations
        self.message_bus = None  # Will be set by orchestrator

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process a user query and generate a response.
        Override in subclasses for agent-specific logic.
        """
        raise NotImplementedError

    async def think(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Agent reasoning step - plan approach before responding.
        """
        return {
            "intent": "unknown",
            "approach": [],
            "required_context": []
        }

    async def act(self, plan: Dict[str, Any]) -> str:
        """
        Execute based on reasoning plan.
        """
        return "Action executed"

    async def communicate(self, message: Message) -> None:
        """
        Send message to another agent via message bus.
        """
        if self.message_bus:
            await self.message_bus.route_message(message)

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "description": self.description
        }


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class PortfolioAgent(Agent):
    """Agent for personal brand, career narrative, achievements."""

    def __init__(self):
        super().__init__(
            agent_id="portfolio",
            name="Portfolio Guru",
            role=AgentRole.PORTFOLIO,
            capabilities=["career_summary", "project_showcase", "skill_highlight", "achievement_narrative"],
            description="Showcases Pranav's career journey, achievements, and personal brand"
        )

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process portfolio-related queries."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["career", "background", "who are you"]):
            return (
                "I'm Pranav, an AI engineer with expertise in cloud operations, "
                "full-stack development, and multi-agent systems. I've built production-grade "
                "systems like obs.atla.in (observability platform) and now atla.in (multi-agent showcase). "
                "My passion is at the intersection of AI, systems design, and startup engineering."
            )

        elif any(word in query_lower for word in ["projects", "built", "showcase"]):
            return (
                "Key projects:\n"
                "1. **obs.atla.in** - Cloud ops platform with real-time metrics, SQLite persistence, "
                "incident simulation\n"
                "2. **atla.in** - Multi-agent AI system with 5 specialized agents\n"
                "3. **AI journey** - 14-week structured learning & building journey\n"
                "Each demonstrates different technical depth: from backend engineering to AI orchestration."
            )

        elif any(word in query_lower for word in ["skills", "expertise", "what can you"]):
            return (
                "Core Skills:\n"
                "• **Backend**: Python, FastAPI, async programming, database design\n"
                "• **Frontend**: TypeScript, React, Next.js, Tailwind CSS\n"
                "• **AI/ML**: Prompt engineering, agent design, RAG systems, LLM integration\n"
                "• **DevOps**: Cloud architecture, observability, incident management\n"
                "• **Full-Stack**: End-to-end system design and implementation"
            )

        else:
            return "Ask me about my career, projects, skills, or achievements!"


class TechnicalAgent(Agent):
    """Agent for deep engineering, architecture, problem-solving."""

    def __init__(self):
        super().__init__(
            agent_id="technical",
            name="Technical Architect",
            role=AgentRole.TECHNICAL,
            capabilities=["architecture_design", "code_review", "system_design", "problem_solving"],
            description="Expert in system architecture, design patterns, and technical problem-solving"
        )

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process technical queries."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["design", "architecture", "scale"]):
            return (
                "To design a scalable system, consider:\n"
                "1. **Separation of Concerns** - Each component has single responsibility\n"
                "2. **Async Operations** - Non-blocking I/O for concurrency\n"
                "3. **Caching Strategy** - Redis for hot data, SQLite for persistence\n"
                "4. **Event-Driven** - Message buses for loose coupling\n"
                "5. **Monitoring** - Real-time metrics, SLO tracking, incident response\n\n"
                "See obs.atla.in for a production example of these principles."
            )

        elif any(word in query_lower for word in ["database", "sql", "data"]):
            return (
                "Database Design Approach:\n"
                "• **Normalization** - Avoid redundancy, use proper indexes\n"
                "• **Query Optimization** - Index on frequently queried columns (timestamp, incident_type)\n"
                "• **Retention Policies** - Auto-cleanup old data (30/90 days)\n"
                "• **Atomicity** - Use transactions for consistency\n"
                "• **Monitoring** - Track slow queries, query plans\n\n"
                "See DATABASE_DESIGN.md in obs.atla.in for detailed schema breakdown."
            )

        elif any(word in query_lower for word in ["agent", "multi-agent", "orchestration"]):
            return (
                "Multi-Agent Architecture:\n"
                "• **Specialization** - Each agent has distinct role (Portfolio, Technical, Strategic, Analytics, Coordinator)\n"
                "• **Message Bus** - Agents communicate via async messages, not direct calls\n"
                "• **Composition** - Complex queries routed through coordinator\n"
                "• **Memory** - Each agent maintains conversation context + long-term memory\n"
                "• **Reasoning** - Agents plan approach before execution\n\n"
                "This enables scalability and clear separation of concerns."
            )

        else:
            return "Ask me about architecture, databases, system design, or multi-agent systems!"


class StrategicAgent(Agent):
    """Agent for planning, decision-making, roadmaps."""

    def __init__(self):
        super().__init__(
            agent_id="strategic",
            name="Strategic Planner",
            role=AgentRole.STRATEGIC,
            capabilities=["roadmap_planning", "priority_setting", "decision_analysis", "goal_setting"],
            description="Helps with strategic planning, roadmaps, and long-term vision"
        )

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process strategic queries."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["roadmap", "plan", "next"]):
            return (
                "14-Week AI Journey Roadmap:\n"
                "**Weeks 1-4**: Foundation (obs.atla.in backend + persistent analytics)\n"
                "**Weeks 5-8**: Multi-agent architecture (5 agents, communication protocol)\n"
                "**Weeks 9-12**: Frontend + Integration (Next.js UI, agent playground)\n"
                "**Weeks 13-14**: Polish & Demo (reasoning transparency, live portfolio)\n\n"
                "Key principle: Build incrementally with clear deliverables at each stage."
            )

        elif any(word in query_lower for word in ["priority", "focus", "what should"]):
            return (
                "Strategic Priorities:\n"
                "1. **Immediate** (Week 6): Backend orchestrator + base agents\n"
                "2. **High** (Week 7-8): Agent communication + memory systems\n"
                "3. **Medium** (Week 9-11): Frontend experiences (playground, showcase)\n"
                "4. **Polish** (Week 12-14): Integration, reasoning transparency, deployment\n\n"
                "Focus on building working prototypes fast, then iterate on quality."
            )

        elif any(word in query_lower for word in ["goal", "achieve", "objective"]):
            return (
                "Key Objectives:\n"
                "✅ Build production-grade multi-agent system (architectural proof)\n"
                "✅ Create compelling portfolio narrative (career/skills proof)\n"
                "✅ Integrate with obs.atla.in for live data (full-stack proof)\n"
                "✅ Demonstrate reasoning and transparency (AI competency proof)\n\n"
                "Success = System that's impressive AND explainable to hiring managers."
            )

        else:
            return "Ask me about roadmaps, priorities, goals, or strategic decisions!"


class AnalyticsAgent(Agent):
    """Agent for insights, metrics, data analysis."""

    def __init__(self):
        super().__init__(
            agent_id="analytics",
            name="Analytics Expert",
            role=AgentRole.ANALYTICS,
            capabilities=["metrics_analysis", "trend_identification", "performance_tracking", "data_visualization"],
            description="Analyzes data, identifies trends, and provides actionable insights"
        )

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process analytics queries."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["metrics", "performance", "data"]):
            return (
                "Key Metrics to Track:\n"
                "• **System Health**: CPU%, Memory%, Latency (p95), Error Rate\n"
                "• **Operational**: Pod Count, RPS, SLO Compliance\n"
                "• **Business**: Conversation Count, Agent Utilization, User Engagement\n"
                "• **Quality**: Response Time, KB Relevance, User Satisfaction\n\n"
                "See obs.atla.in analytics endpoints for real-time data."
            )

        elif any(word in query_lower for word in ["trend", "insight", "pattern"]):
            return (
                "Key Insights from obs.atla.in:\n"
                "• **Incident Patterns**: Traffic spikes cause cascading failures (CPU→Pods→Latency)\n"
                "• **Recovery Speed**: Smooth decay takes ~15 seconds with exponential factor\n"
                "• **SLO Impact**: Peak latency breaches SLO; remediation takes time\n"
                "• **Agent Effectiveness**: KB source retrieval improves answer quality\n\n"
                "These patterns inform multi-agent system design."
            )

        else:
            return "Ask me about metrics, trends, performance analysis, or data insights!"


class CoordinatorAgent(Agent):
    """Agent that routes requests to other agents and synthesizes responses."""

    def __init__(self):
        super().__init__(
            agent_id="coordinator",
            name="Coordinator",
            role=AgentRole.COORDINATOR,
            capabilities=["intent_routing", "multi_agent_orchestration", "response_synthesis", "context_management"],
            description="Routes queries to appropriate agents and synthesizes multi-agent responses"
        )

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Route query to appropriate agent(s)."""
        # This is a simplified version; in production, would use LLM for routing
        query_lower = query.lower()

        routing_map = {
            "portfolio": ["career", "background", "projects", "skills", "achievement"],
            "technical": ["design", "architecture", "code", "database", "system"],
            "strategic": ["roadmap", "plan", "next", "priority", "goal"],
            "analytics": ["metrics", "data", "trend", "insight", "performance"]
        }

        # Simple keyword-based routing (in production, use LLM)
        for agent_id, keywords in routing_map.items():
            if any(kw in query_lower for kw in keywords):
                return f"[Routing to {agent_id} agent...]\n\nThis would delegate to the appropriate specialized agent."

        return "I'm not sure which agent to route this to. Could you be more specific?"


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class MessageBus:
    """Manages inter-agent communication."""

    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()

    def register_agent(self, agent: Agent):
        """Register agent with message bus."""
        self.agents[agent.agent_id] = agent

    async def route_message(self, message: Message):
        """Route message to recipient agent or broadcast."""
        await self.message_queue.put(message)

    async def process_messages(self):
        """Process messages from queue."""
        while True:
            message = await self.message_queue.get()
            # In production, route to specific agent or broadcast
            self.message_queue.task_done()


class Orchestrator:
    """Orchestrates multi-agent system."""

    def __init__(self):
        self.agents = []
        self.message_bus = MessageBus()
        self._initialize_agents()

    def _initialize_agents(self):
        """Create all specialized agents."""
        self.agents = [
            PortfolioAgent(),
            TechnicalAgent(),
            StrategicAgent(),
            AnalyticsAgent(),
            CoordinatorAgent()
        ]

        # Register agents with message bus
        for agent in self.agents:
            agent.message_bus = self.message_bus
            self.message_bus.register_agent(agent)

    async def process_query(self, request: ChatRequest) -> ChatResponse:
        """
        Process user query through appropriate agent(s).
        If agent_id specified, use that agent.
        Otherwise, coordinator routes it.
        """
        if request.agent_id:
            # Direct query to specific agent
            agent = next((a for a in self.agents if a.agent_id == request.agent_id), None)
            if not agent:
                raise ValueError(f"Agent {request.agent_id} not found")
        else:
            # Use coordinator for routing
            agent = next((a for a in self.agents if a.role == AgentRole.COORDINATOR), None)

        response = await agent.process_query(request.query, {})

        return ChatResponse(
            conversation_id=request.conversation_id or f"conv_{datetime.utcnow().timestamp()}",
            response=response,
            agent_id=agent.agent_id,
            agents_involved=[agent.agent_id],
            reasoning_chain=None,
            timestamp=datetime.utcnow()
        )

    def get_agents(self) -> List[Dict[str, Any]]:
        """Return list of all agents with metadata."""
        return [agent.get_info() for agent in self.agents]


# ============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# ============================================================================

orchestrator = Orchestrator()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "agents": len(orchestrator.agents)
    }


@app.get("/agents")
async def list_agents():
    """Get list of all available agents."""
    return {
        "agents": orchestrator.get_agents(),
        "total": len(orchestrator.agents)
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for user queries.
    Routes to appropriate agent(s) for response.
    """
    try:
        response = await orchestrator.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{agent_id}")
async def chat_agent(agent_id: str, request: ChatRequest) -> ChatResponse:
    """
    Direct query to specific agent.
    """
    request.agent_id = agent_id
    try:
        response = await orchestrator.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    agent = next((a for a in orchestrator.agents if a.agent_id == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent.get_info()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("🚀 AtlaOps Multi-Agent System starting...")
    print(f"📦 Initialized {len(orchestrator.agents)} agents:")
    for agent in orchestrator.agents:
        print(f"  - {agent.name} ({agent.agent_id})")

    uvicorn.run(app, host="0.0.0.0", port=8001)
