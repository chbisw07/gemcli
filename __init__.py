# gemcli/__init__.py

__version__ = "0.1.0"

# For external import access (e.g., from Streamlit or plugins)
from main import main
from agent import Agent
from models import ModelRegistry, LLMModel
from tools.registry import ToolRegistry
from tools.direct_parser import try_direct_actions
