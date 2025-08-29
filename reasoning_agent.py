# reasoning_agent.py
import json
import re  # Added missing import
from typing import List, Dict, Any
from agent import Agent
from tools.registry import ToolRegistry
from models import LLMModel

class ReasoningAgent(Agent):
    """
    Enhanced agent with multi-step reasoning capabilities
    """
    def __init__(self, model, tools, enable_tools=True):
        super().__init__(model, tools, enable_tools)
    
    def analyze_and_plan(self, query: str) -> list:
        """
        Analyze the query and create an execution plan with multiple steps
        """
        planning_prompt = f"""
        Analyze this coding request and create a step-by-step execution plan:
        {query}
        
        Available tools: {list(self.tools.tools.keys())}
        
        Return a JSON array of steps, each with:
        - tool: tool name to use
        - args: arguments for the tool
        - description: what this step accomplishes
        
        Focus on:
        1. First understanding the codebase structure
        2. Finding relevant files
        3. Analyzing the code
        4. Implementing changes
        5. Validating the solution
        """
        
        messages = [
            {"role": "system", "content": "You are an expert coding assistant that creates detailed execution plans."},
            {"role": "user", "content": planning_prompt}
        ]
        
        try:
            resp = self.adapter.chat(messages)
            content = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "[]")
            
            # Extract JSON from response (might be wrapped in markdown)
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            plan = json.loads(content)
            return plan
        except Exception as e:
            return [{"error": f"Failed to create plan: {str(e)}"}]
    
    def execute_plan(self, plan: list, max_iters: int = 10) -> str:
        """
        Execute the planned steps with proper error handling
        """
        results = []
        
        for i, step in enumerate(plan):
            if "error" in step:
                results.append(step)
                continue
                
            try:
                # Execute the step
                result = self.tools.call(step['tool'], **step.get('args', {}))
                results.append({
                    'step': i + 1,
                    'description': step.get('description', ''),
                    'tool': step['tool'],
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'step': i + 1,
                    'description': step.get('description', ''),
                    'tool': step['tool'],
                    'error': str(e),
                    'success': False
                })
                # Don't continue if a critical step fails
                if step.get('critical', True):
                    break
        
        return json.dumps(results, indent=2)
    
    def ask_with_planning(self, query: str, max_iters: int = 10) -> str:
        """
        Main method that plans and executes complex queries
        """
        # First try direct execution for simple queries
        direct = self._try_direct_actions(query)
        if direct is not None:
            return direct
            
        # For complex queries, use planning
        plan = self.analyze_and_plan(query)
        return self.execute_plan(plan, max_iters)