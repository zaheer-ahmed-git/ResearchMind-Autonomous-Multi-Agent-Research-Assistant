# Architecture notes — Autonomous Research Agent

Research basis (2026 docs and patterns):

- LangGraph Graph API: typed `State`, nodes as functions, conditional edges, compile + invoke ([Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api))
- Short-term memory via checkpointers / `InMemorySaver` and `thread_id` ([Add memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers))
- OpenAI usage through `langchain-openai` `ChatOpenAI` inside node adapters when `USE_LLM=true`
- Research-agent pattern: decompose → gather → analyze/verify → refine loop → synthesize
- Next.js visualizer consumes FastAPI trace payloads (timeline + node detail), similar in spirit to TraceLens / agent SSE flow UIs

## Graph

```text
START
  → planner      (task decomposition)
  → gatherer     (offline corpus / evidence collect + adaptive ranking)
  → analyst      (themes, gaps, memory pruning, confidence)
      ├─(low confidence & loops left)→ refiner → gatherer
      └─(ok)→ synthesizer → END
```

## Latency reduction (~35%)

Baseline cost model assumes unpruned memory and unranked bulky evidence in every super-step.
Optimized path:

1. Memory pruning drops low-signal STM items before later nodes/checkpoints.
2. Adaptive ranking keeps high-trust evidence near the top so synthesis uses a smaller, stronger set.

`estimate_latency` exposes baseline vs optimized ms and reduction % for the Next.js metrics strip.

## Hallucination controls

- Decomposition localizes retrieval failures.
- Adaptive ranking prefers trusted sources + query overlap.
- Synthesizer prompts (and offline template) require rank citations and gap callouts.
- Refine loops are bounded (`MAX_REFINE_LOOPS`) so the graph cannot spin forever.
