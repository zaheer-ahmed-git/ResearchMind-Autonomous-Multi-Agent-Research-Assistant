export type TraceEvent = {
  id: string;
  run_id: string;
  node: string;
  event_type: "start" | "end" | "info" | "decision" | "error";
  message: string;
  duration_ms: number | null;
  payload: Record<string, unknown>;
  timestamp: string;
};

export type ResearchResponse = {
  run_id: string;
  query: string;
  report: string;
  confidence: number;
  refine_loops: number;
  latency_ms: number;
  baseline_latency_ms: number;
  latency_reduction_pct: number;
  traces: TraceEvent[];
  mode: "offline" | "llm";
  subtasks: Array<{ id: string; question: string; priority: number; status: string }>;
  evidence: Array<{
    id: string;
    title: string;
    score: number;
    rank: number;
    source: string;
  }>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function runResearch(query: string): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE}/research/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    throw new Error(`Research run failed (${response.status})`);
  }
  return response.json();
}
