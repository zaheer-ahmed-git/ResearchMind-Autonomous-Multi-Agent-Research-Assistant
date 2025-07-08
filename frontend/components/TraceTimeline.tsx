"use client";

import type { TraceEvent } from "@/lib/api";

const NODE_COLORS: Record<string, string> = {
  planner: "#1d4f7c",
  gatherer: "#2f6f4e",
  analyst: "#8a5a1e",
  refiner: "#7a3e5c",
  synthesizer: "#3d3a7a",
};

type Props = {
  traces: TraceEvent[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function TraceTimeline({ traces, selectedId, onSelect }: Props) {
  return (
    <ol className="timeline">
      {traces.map((event) => {
        const color = NODE_COLORS[event.node] ?? "#445";
        const active = event.id === selectedId;
        return (
          <li key={event.id}>
            <button
              type="button"
              className={active ? "trace-item active" : "trace-item"}
              style={{ borderLeftColor: color }}
              onClick={() => onSelect(event.id)}
            >
              <span className="trace-meta">
                <strong>{event.node}</strong>
                <em>{event.event_type}</em>
                {event.duration_ms != null ? <span>{event.duration_ms} ms</span> : null}
              </span>
              <span className="trace-message">{event.message}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
