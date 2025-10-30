# Architecture notes — Autonomous Research Agent

Research basis (2026 docs and patterns):

- LangGraph Graph API: typed `State`, nodes as functions, conditional edges, compile + invoke ([Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api))
- Short-term memory via checkpointers / `InMemorySaver` and `thread_id` ([Add memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers))
- OpenAI usage through `langchain-openai` `ChatOpenAI` inside node adapters when `USE_LLM=true`
- Research-agent pattern: decompose → gather → analyze/verify → refine loop → synthesize
- Next.js visualizer consumes FastAPI trace payloads (timeline + node detail), similar in spirit to TraceLens / agent SSE flow UIs

## Step-by-step system

The pipeline consolidates the original Agent Network, Human-in-the-loop, and Memory modules.

```text
START
  → coordinator          (Agent Network: decompose + route)
  → research             (ResearchAgent: gather, rank, write STM + vector memory)
  → analysis             (AnalysisAgent: themes, gaps, prune STM, vector search)
      ├─ gaps remain → coordinator refine → research
      └─ ready → synthesis   (SynthesisAgent: grounded report, stored in vector memory)
  → qa                   (QAAgent: score, weaknesses, needs-human flag)
  → human_checkpoint     (HumanCheckpoint + verification)
      ├─ pause_for_human and low QA → LangGraph interrupt, resume with Command
      ├─ rejected once → analysis
      └─ approved → END
```

| Original module | Where it lives now |
|---|---|
| `Agent Network/CoordinatorAgent` | `agents/coordinator.py` |
| `Agent Network/ResearchAgent` | `agents/research.py` |
| `Agent Network/AnalysisAgent` | `agents/analysis.py` |
| `Agent Network/SynthesisAgent` | `agents/synthesis.py` |
| `Agent Network/QualityAssuranceAgent` | `agents/qa.py` |
| `HumaninLoop.py/human_checkpoint.py` | `hitl/checkpoint.py` |
| `HumaninLoop.py/human_verification.py` | `hitl/verification.py` |
| `Memory/Stm.py` | `memory/stm.py` |
| `Memory/VectorDatabase.py` | `memory/vector_store.py` (same document metadata, local index) |

Human review uses LangGraph `interrupt()` and `Command(resume=...)` with a checkpointer and stable `thread_id` (the run id). See the [interrupts guide](https://docs.langchain.com/oss/python/langgraph/interrupts).

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
