"use client";

import { FormEvent, useMemo, useState } from "react";
import { ResearchResponse, runResearch } from "@/lib/api";
import { TraceTimeline } from "@/components/TraceTimeline";

export function RunPanel() {
  const [query, setQuery] = useState(
    "How do LangGraph research agents reduce hallucinations with memory pruning?"
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selected = useMemo(
    () => result?.traces.find((event) => event.id === selectedId) ?? null,
    [result, selectedId]
  );

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await runResearch(query.trim());
      setResult(data);
      setSelectedId(data.traces[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="layout">
      <section className="hero-panel">
        <p className="brand">Autonomous Research Agent</p>
        <h1>Execution trace visualizer</h1>
        <p className="lede">
          Watch LangGraph nodes decompose, gather, prune, rank, refine, and synthesize.
        </p>
        <form onSubmit={onSubmit} className="run-form">
          <label htmlFor="query">Research query</label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            required
          />
          <button type="submit" disabled={loading || query.trim().length < 3}>
            {loading ? "Running graph…" : "Start research run"}
          </button>
        </form>
        {error ? <p className="error">{error}</p> : null}
      </section>

      {result ? (
        <section className="results">
          <div className="metrics">
            <div>
              <span>Latency</span>
              <strong>{result.latency_ms.toFixed(1)} ms</strong>
            </div>
            <div>
              <span>Baseline</span>
              <strong>{result.baseline_latency_ms.toFixed(1)} ms</strong>
            </div>
            <div>
              <span>Reduction</span>
              <strong>{result.latency_reduction_pct.toFixed(1)}%</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{(result.confidence * 100).toFixed(0)}%</strong>
            </div>
          </div>

          <div className="split">
            <div>
              <h2>Trace timeline</h2>
              <TraceTimeline
                traces={result.traces}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>
            <div>
              <h2>Node detail</h2>
              {selected ? (
                <pre className="detail">{JSON.stringify(selected, null, 2)}</pre>
              ) : (
                <p className="muted">Select a trace event.</p>
              )}
              <h2>Report</h2>
              <pre className="report">{result.report}</pre>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
