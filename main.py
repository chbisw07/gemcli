# main_enhanced.py
import argparse
import re
import json
from dotenv import load_dotenv

# Auto-load .env early so api keys are available to adapters
load_dotenv()

from models import ModelRegistry
from tools.registry import ToolRegistry
from agent import Agent

# Try to import enhanced components, but fall back to standard ones
try:
    from tools.enhanced_registry import EnhancedToolRegistry
    from reasoning_agent import ReasoningAgent
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    # Create dummy classes for fallback
    class EnhancedToolRegistry:
        def __init__(self, project_root):
            from tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry(project_root)
            self.tools = self.tool_registry.tools
    
    class ReasoningAgent(Agent):
        def ask_with_planning(self, query, max_iters=10):
            return self.ask_once(query, max_iters)


def main():
    parser = argparse.ArgumentParser(
        description="gemcli - Enhanced CLI assistant for local codebases with multi-step reasoning"
    )
    parser.add_argument("--config", default="models.json", help="Path to model config JSON")
    parser.add_argument("--model", help="Model name to use (optional; defaults from config)")
    parser.add_argument("--root", required=True, help="Project root directory")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool use")
    parser.add_argument("--query", help="Single-shot question to answer (optional)")
    parser.add_argument("--max-iters", type=int, default=5, help="Max tool iterations")
    parser.add_argument("--plan", action="store_true", 
                        help="Use multi-step planning for complex queries")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use enhanced tools and reasoning capabilities")
    
    args = parser.parse_args()

    models = ModelRegistry(args.config)
    model = models.get(args.model)  # picks default if None

    if args.enhanced and not ENHANCED_AVAILABLE:
        print("Warning: Enhanced features not available. Using standard mode.")
        args.enhanced = False

    if args.enhanced:
        tools = EnhancedToolRegistry(args.root)
        agent = ReasoningAgent(model, tools, enable_tools=not args.no_tools)
    else:
        tools = ToolRegistry(args.root)
        agent = Agent(model, tools, enable_tools=not args.no_tools)

    if args.query:
        try:
            if args.plan and args.enhanced:
                result = agent.ask_with_planning(args.query, max_iters=args.max_iters)
            else:
                result = agent.ask_once(args.query, max_iters=args.max_iters)
            print(result)
        except Exception as e:
            print(f"[error] {e}")
        return

    # REPL mode
    tool_names = list(getattr(tools, "tools", {}).keys())
    mode = "enhanced" if args.enhanced else "standard"
    planning = "with planning" if args.plan else "without planning"
    
    print(f"Model: {model.name} ({model.provider}) | Mode: {mode} {planning}")
    print(f"Tools: {', '.join(tool_names) if tool_names else 'disabled'}")
    print("Enter your question (Ctrl+C to exit)")
    
    while True:
        try:
            q = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("bye!")
            break
        if not q:
            continue
        try:
            if args.plan and args.enhanced:
                ans = agent.ask_with_planning(q, max_iters=args.max_iters)
            else:
                ans = agent.ask_once(q, max_iters=args.max_iters)
        except Exception as e:
            ans = f"[error] {e}"
        print(f"assistant> {ans}")


if __name__ == "__main__":
    main()