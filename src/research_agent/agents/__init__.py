"""Agent network for the step-by-step research system.

Stages, in order:
1. CoordinatorAgent — decompose the query and route the next stage
2. ResearchAgent — gather and rank evidence
3. AnalysisAgent — themes, gaps, confidence
4. SynthesisAgent — grounded report
5. QAAgent — validation score and human-review flag
"""

from research_agent.agents.analysis import AnalysisAgent
from research_agent.agents.coordinator import CoordinatorAgent
from research_agent.agents.qa import QAAgent
from research_agent.agents.research import ResearchAgent
from research_agent.agents.synthesis import SynthesisAgent

__all__ = [
    "AnalysisAgent",
    "CoordinatorAgent",
    "QAAgent",
    "ResearchAgent",
    "SynthesisAgent",
]
