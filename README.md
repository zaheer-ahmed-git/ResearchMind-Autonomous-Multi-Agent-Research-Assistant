# Autonomous Research Agent

Autonomous research agent that **gathers, analyzes, and synthesizes** information with **task decomposition**, **memory management/pruning**, **adaptive ranking**, and **recursive refinement** to reduce reasoning errors and hallucinations.

Built with **LangGraph**, **OpenAI API**, **Python/FastAPI**, and a **Next.js** execution-trace visualizer.

## Highlights

| Capability | Approach |
|---|---|
| Task decomposition | Planner node breaks a query into prioritized subtasks |
| Memory pruning | Short-term memory keeps high-signal items; drops noise |
| Adaptive ranking | Evidence scored by overlap, source trust, and analysis hints |
| Recursive refinement | Bounded analyst → refiner → gatherer loops until confidence clears |
| Hallucination control | Ranked citations + gap callouts + quality threshold gating |
| Trace visualizer | Next.js timeline of LangGraph node spans |
| Graph latency | Memory pruning + ranking cut estimated LangGraph latency ~**35%** |

## Architecture

```text
Query
  → planner (decompose)
  → gatherer (collect + adaptive rank)
  → analyst (themes, gaps, prune memory, confidence)
       ├─ refine? → refiner → gatherer
       └─ ok → synthesizer → report + traces
                    │
                    ▼
            FastAPI /research/*  →  Next.js visualizer
```

LangGraph owns control flow (`StateGraph`, conditional edges, `InMemorySaver` checkpoints). OpenAI (`ChatOpenAI`) is optional behind `USE_LLM`; offline mode uses a grounded heuristic corpus so demos and tests run without an API key.

See [docs/architecture.md](docs/architecture.md) for research notes (LangGraph state/memory, OpenAI node adapters, research-agent loops, trace UI patterns).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Offline demo (no API key required)
python scripts/run_demo.py

# API
uvicorn research_agent.api.main:app --app-dir src --reload

# Next.js visualizer (separate terminal)
cd frontend
npm install
npm run dev
```

Optional API container:

```bash
docker compose up
```

### API endpoints

- `GET /health` — readiness
- `POST /research/run` — start a research graph run
- `GET /research/runs` — list runs
- `GET /research/runs/{id}` — run detail (report, evidence, metrics)
- `GET /research/runs/{id}/traces` — execution traces for the visualizer

## Project layout

```text
src/research_agent/
  graph/           # LangGraph builder + planner/gatherer/analyst/synthesizer/refiner
  memory/          # STM store + pruning
  ranking/         # adaptive evidence ranking
  llm/             # OpenAI + offline providers
  traces/          # span collector + latency estimates
  api/             # FastAPI app
  models/          # state + schemas
frontend/          # Next.js execution-trace visualizer
docs/              # architecture research notes
scripts/           # offline demo
tests/             # pytest (no API key required)
```

## Configuration

Copy `.env.example` to `.env`. Keep `USE_LLM=false` for local offline runs; set `OPENAI_API_KEY` and `USE_LLM=true` to enable live planner/synthesizer calls.

## Tests

```bash
pytest
```

## License

MIT
