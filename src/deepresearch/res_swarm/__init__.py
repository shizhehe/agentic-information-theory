"""
ResSwarm - Research Swarm Intelligence

A provider-agnostic supervisor-worker architecture for evaluating LLM pairs 
on complex research tasks using the DeepResearch Bench benchmark.
"""

__version__ = "1.0.0"
__author__ = "ResSwarm Team"
__description__ = "Research Swarm Intelligence - Supervisor-Worker LLM Evaluation"

# Import main components for easy access
from src.deepresearch.res_swarm.src.deepres_minion import DeepResearchMinion
from src.deepresearch.res_swarm.src.deepres_tools import WebSearchTool
from src.deepresearch.res_swarm.configs.model_config import get_config_manager

__all__ = [
    "DeepResearchMinion",
    "WebSearchTool", 
    "get_config_manager"
]