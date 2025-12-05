from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MiniLab.llm_backends.base import ChatMessage, LLMBackend


# Maximum iterations for agentic loops to prevent runaway
MAX_AGENTIC_ITERATIONS = 15


@dataclass
class Agent:
    id: str
    display_name: str
    guild: str
    role: str
    persona: str
    backend: LLMBackend
    tools: List[str] = field(default_factory=list)
    tool_instances: Dict[str, Any] = field(default_factory=dict)  # Actual tool objects
    colleagues: Dict[str, 'Agent'] = field(default_factory=dict)  # Other agents for consultation

    # In future this could be a proper memory object / RAG store
    memory_notes: List[str] = field(default_factory=list)
    
    def set_colleagues(self, agents: Dict[str, 'Agent']):
        """Set the dictionary of colleague agents for cross-consultation."""
        self.colleagues = {k: v for k, v in agents.items() if k != self.id}
    
    async def ask_colleague(
        self, 
        colleague_id: str, 
        question: str,
        context: str = "",
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """
        Ask another agent for input or advice on a specific point.
        
        Use this when you need:
        - Clarification on a suggestion from another agent
        - Quick statistical/methodological advice
        - Domain expertise outside your specialty
        - Verification of an approach before implementing
        
        Args:
            colleague_id: The ID of the agent to consult (e.g., "bayes", "feynman")
            question: Your specific question for them
            context: Brief context about what you're working on
            max_tokens: Maximum response length
            
        Returns:
            Dict with 'success', 'response', and 'colleague' fields
        """
        if colleague_id not in self.colleagues:
            available = list(self.colleagues.keys())
            return {
                "success": False,
                "error": f"Unknown colleague: {colleague_id}",
                "available_colleagues": available,
                "hint": f"Available colleagues: {', '.join(available)}",
            }
        
        colleague = self.colleagues[colleague_id]
        
        # Format the consultation request
        consultation_prompt = f"""A colleague ({self.display_name}, {self.role}) is asking for your input:

CONTEXT: {context if context else 'Working on current project analysis'}

QUESTION: {question}

Please provide a focused, helpful response. Be concise but thorough."""

        try:
            response = await colleague.arespond(
                consultation_prompt,
                max_tokens=max_tokens,
            )
            
            return {
                "success": True,
                "colleague": colleague.display_name,
                "colleague_role": colleague.role,
                "question": question,
                "response": response,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error consulting {colleague.display_name}: {str(e)}",
                "colleague": colleague_id,
            }
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if agent has access to a specific tool."""
        return tool_name in self.tool_instances
    
    async def use_tool(self, tool_name: str, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool if available."""
        if not self.has_tool(tool_name):
            error_msg = (
                f"TOOL ACCESS ERROR: {self.display_name} attempted to use '{tool_name}' but doesn't have access.\n"
                f"Available tools: {list(self.tool_instances.keys())}\n"
                f"Action attempted: {action}\n"
                f"Parameters: {kwargs}"
            )
            print(f"\n❌ {error_msg}\n")
            return {
                "success": False,
                "error": error_msg,
                "available_tools": list(self.tool_instances.keys()),
            }
        
        tool = self.tool_instances[tool_name]
        try:
            result = await tool.execute(action=action, **kwargs)
            if not result.get("success", False):
                error_msg = (
                    f"TOOL EXECUTION FAILED: {self.display_name} used '{tool_name}' but it failed.\n"
                    f"Action: {action}\n"
                    f"Error: {result.get('error', 'Unknown error')}\n"
                    f"Parameters: {kwargs}"
                )
                print(f"\n❌ {error_msg}\n")
            return result
        except Exception as e:
            error_msg = (
                f"TOOL EXCEPTION: {self.display_name} encountered an exception using '{tool_name}'.\n"
                f"Action: {action}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Parameters: {kwargs}"
            )
            print(f"\n❌ {error_msg}\n")
            return {
                "success": False,
                "error": error_msg,
                "exception": str(e),
            }

    async def arespond(
        self,
        user_message: str,
        context: str | None = None,
        max_tokens: int = 3000,
    ) -> str:
        """
        Basic response method: persona + optional context + user message.
        
        Args:
            user_message: The question or prompt for the agent
            context: Optional shared context
            max_tokens: int = 3000
        """
        # Team roster for all agents
        team_roster = (
            "MiniLab Team Structure:\n"
            "Synthesis Guild: Bohr (project lead, computational oncologist), Farber (clinical pathologist, adversarial critic), Gould (librarian, science writer)\n"
            "Theory Guild: Feynman (theoretical physicist, creative thinker), Shannon (information theorist, causal designer), Greider (molecular biologist, mechanistic expert)\n"
            "Implementation Guild: Bayes (Bayesian statistician, clinical trials), Hinton (computer scientist, infrastructure guru), Dayhoff (bioinformatician, biochemist)\n\n"
            "The User is a fellow scientist and expert in their domain. Communicate as you would with a scientific colleague."
        )
        
        # Colleague consultation info
        colleague_info = ""
        if self.colleagues:
            colleague_list = [f"{c.display_name} ({c.role})" for c in self.colleagues.values()]
            colleague_info = (
                "\n\nCOLLEAGUE CONSULTATION:\n"
                "You can pause your work to ask any colleague for quick input or advice. "
                "Available colleagues: " + ", ".join(colleague_list) + "\n"
                "Use this when you need clarification, domain expertise, or verification of an approach."
            )
        
        messages: List[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    f"You are {self.display_name}, {self.role} in the MiniLab team.\n\n"
                    f"{team_roster}\n\n"
                    f"Your Persona:\n{self.persona}\n\n"
                    "CRITICAL OPERATING MODE:\n"
                    "- You are ACTIVELY WORKING on a project RIGHT NOW, not planning for the future\n"
                    "- When you say 'we will do X', it means 'we are doing X in the next immediate step'\n"
                    "- Do not reference timelines like 'over the coming days/weeks' - everything happens NOW\n"
                    "- Focus on IMMEDIATE ACTIONS that can be executed in this session\n"
                    "- Be specific about what file/script/analysis is being created NEXT\n\n"
                    "FILE ACCESS:\n"
                    "- ReadData/: READ-ONLY access (explore and read source data)\n"
                    "- Sandbox/: READ-WRITE access (create scripts, outputs, scratch files)\n"
                    "- All scripts must read from ReadData/ and write to Sandbox/\n\n"
                    "Guidelines:\n"
                    "- Be concise and focused - say what you need to say and no more\n"
                    "- Ground scientific claims in cited literature when relevant\n"
                    "- Aim for clarity over verbosity\n"
                    "- Typical responses should be 2-4 paragraphs unless more detail is specifically requested\n"
                    "- Remember: you're speaking with fellow experts, not laypeople\n\n"
                    f"Available tools: {', '.join(self.tool_instances.keys()) if self.tool_instances else 'None'}"
                    f"{colleague_info}"
                ),
            },
        ]
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Shared context for this discussion:\n{context}",
                }
            )
        messages.append({"role": "user", "content": user_message})

        reply = await self.backend.acomplete(messages, max_tokens=max_tokens)
        # Later: update memory based on reply, etc.
        return reply

    async def arespond_with_vision(
        self,
        user_message: str,
        image_paths: Optional[List[str]] = None,
        pdf_path: Optional[str] = None,
        context: str | None = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Response method with vision capability - agent can see images/PDFs.
        
        Args:
            user_message: The question or prompt for the agent
            image_paths: List of paths to image files (PNG, JPEG, etc.)
            pdf_path: Path to a PDF file (will be converted to images)
            context: Optional shared context
            max_tokens: Maximum response tokens
        """
        # Check if backend supports vision
        if not hasattr(self.backend, 'acomplete_with_vision'):
            print(f"  ⚠ {self.display_name}'s backend doesn't support vision. Using text-only.")
            return await self.arespond(user_message, context, max_tokens)
        
        # Import vision helpers
        try:
            from MiniLab.llm_backends.anthropic_backend import pdf_to_images, image_to_base64
        except ImportError:
            print("  ⚠ Vision helpers not available. Using text-only.")
            return await self.arespond(user_message, context, max_tokens)
        
        # Collect images
        images = []
        
        # Convert PDF to images if provided
        if pdf_path:
            pdf_images = pdf_to_images(pdf_path, max_pages=6)  # Limit to 6 pages
            if pdf_images:
                images.extend(pdf_images)
                print(f"  📄 {self.display_name} viewing PDF ({len(pdf_images)} pages)")
            else:
                print(f"  ⚠ Could not convert PDF for {self.display_name}")
        
        # Add individual images
        if image_paths:
            for img_path in image_paths:
                img_data = image_to_base64(img_path)
                if img_data:
                    images.append(img_data)
            if images:
                print(f"  🖼️ {self.display_name} viewing {len(image_paths)} image(s)")
        
        if not images:
            print(f"  ⚠ No images available for {self.display_name}. Using text-only.")
            return await self.arespond(user_message, context, max_tokens)
        
        # Build messages
        team_roster = (
            "MiniLab Team Structure:\n"
            "Synthesis Guild: Bohr (project lead, computational oncologist), Farber (clinical pathologist, adversarial critic), Gould (librarian, science writer)\n"
            "Theory Guild: Feynman (theoretical physicist, creative thinker), Shannon (information theorist, causal designer), Greider (molecular biologist, mechanistic expert)\n"
            "Implementation Guild: Bayes (Bayesian statistician, clinical trials), Hinton (computer scientist, infrastructure guru), Dayhoff (bioinformatician, biochemist)\n\n"
            "The User is a fellow scientist and expert in their domain."
        )
        
        messages: List[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    f"You are {self.display_name}, {self.role} in the MiniLab team.\n\n"
                    f"{team_roster}\n\n"
                    f"Your Persona:\n{self.persona}\n\n"
                    "Guidelines:\n"
                    "- You are viewing images/figures. Describe what you see in detail.\n"
                    "- Be specific about visual elements: colors, labels, axes, data points, patterns.\n"
                    "- Provide scientific interpretation of visualizations.\n"
                    "- Note any issues with clarity, labeling, or presentation.\n"
                ),
            },
        ]
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Shared context for this discussion:\n{context}",
                }
            )
        messages.append({"role": "user", "content": user_message})

        reply = await self.backend.acomplete_with_vision(
            messages, 
            images=images, 
            max_tokens=max_tokens
        )
        return reply

    async def agentic_execute(
        self,
        task: str,
        context: str = "",
        max_iterations: int = MAX_AGENTIC_ITERATIONS,
        max_tokens_per_step: int = 2000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a task with full agentic capabilities (ReAct-style loop).
        
        The agent can:
        - Think about what to do next
        - Use tools and see results
        - Consult colleagues and incorporate their input
        - Iterate until the task is complete
        
        This is the TRUE agentic mode - like a VS Code agent.
        
        Args:
            task: The task to accomplish
            context: Additional context
            max_iterations: Safety limit on iterations
            max_tokens_per_step: Tokens per reasoning step
            verbose: Print progress
            
        Returns:
            Dict with 'success', 'result', 'iterations', 'tool_calls', etc.
        """
        # Build tool descriptions
        tool_descriptions = self._build_tool_descriptions()
        colleague_descriptions = self._build_colleague_descriptions()
        
        system_prompt = f"""You are {self.display_name}, {self.role} in the MiniLab team.

{self.persona}

You are operating in AGENTIC MODE. You can:
1. THINK about what to do
2. USE TOOLS to take actions  
3. ASK COLLEAGUES for help
4. OBSERVE results and continue

AVAILABLE TOOLS:
{tool_descriptions}

AVAILABLE COLLEAGUES:
{colleague_descriptions}

HOW TO USE TOOLS:
When you want to use a tool, output a JSON block like this:
```tool
{{"tool": "tool_name", "action": "action_name", "params": {{"key": "value"}}}}
```

HOW TO ASK A COLLEAGUE:
```colleague
{{"colleague": "agent_id", "question": "Your question here"}}
```

HOW TO SIGNAL COMPLETION:
When you've finished the task, output:
```done
{{"result": "Summary of what you accomplished", "outputs": ["list", "of", "files", "created"]}}
```

WORKFLOW:
1. Think step by step about what you need to do
2. Use tools/colleagues as needed
3. Observe results and adjust
4. When complete, signal done

Be concise. Act, don't just describe what you would do.
"""
        
        messages: List[ChatMessage] = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        
        messages.append({"role": "user", "content": f"TASK: {task}"})
        
        # Tracking
        iterations = 0
        tool_calls = []
        colleague_calls = []
        all_outputs = []
        
        while iterations < max_iterations:
            iterations += 1
            
            if verbose:
                print(f"      [{self.display_name}] Iteration {iterations}...")
            
            # Get agent's response
            response = await self.backend.acomplete(messages, max_tokens=max_tokens_per_step)
            
            # Add response to conversation
            messages.append({"role": "assistant", "content": response})
            
            # Check for completion
            done_match = re.search(r'```done\s*\n(.*?)\n```', response, re.DOTALL)
            if done_match:
                try:
                    done_data = json.loads(done_match.group(1))
                    if verbose:
                        print(f"      [{self.display_name}] ✓ Task complete")
                    return {
                        "success": True,
                        "result": done_data.get("result", response),
                        "outputs": done_data.get("outputs", all_outputs),
                        "iterations": iterations,
                        "tool_calls": tool_calls,
                        "colleague_calls": colleague_calls,
                        "final_response": response,
                    }
                except json.JSONDecodeError:
                    pass  # Continue if JSON is malformed
            
            # Check for tool calls
            tool_match = re.search(r'```tool\s*\n(.*?)\n```', response, re.DOTALL)
            if tool_match:
                try:
                    tool_data = json.loads(tool_match.group(1))
                    tool_name = tool_data.get("tool")
                    action = tool_data.get("action")
                    params = tool_data.get("params", {})
                    
                    if verbose:
                        print(f"        🔧 Using {tool_name}.{action}...")
                    
                    # Execute the tool
                    result = await self.use_tool(tool_name, action, **params)
                    tool_calls.append({
                        "tool": tool_name,
                        "action": action,
                        "params": params,
                        "result": result,
                    })
                    
                    # Format result for agent
                    result_str = json.dumps(result, indent=2, default=str)
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + "\n... (truncated)"
                    
                    # Add observation to conversation
                    messages.append({
                        "role": "user",
                        "content": f"TOOL RESULT ({tool_name}.{action}):\n```\n{result_str}\n```\n\nContinue with your task."
                    })
                    
                    if result.get("success"):
                        if "path" in result:
                            all_outputs.append(result["path"])
                    
                    continue  # Get next response
                    
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "user",
                        "content": f"ERROR: Could not parse tool call JSON: {e}. Please use valid JSON format."
                    })
                    continue
            
            # Check for colleague consultation
            colleague_match = re.search(r'```colleague\s*\n(.*?)\n```', response, re.DOTALL)
            if colleague_match:
                try:
                    colleague_data = json.loads(colleague_match.group(1))
                    colleague_id = colleague_data.get("colleague")
                    question = colleague_data.get("question")
                    
                    if verbose:
                        print(f"        💬 Asking {colleague_id}...")
                    
                    # Ask the colleague
                    result = await self.ask_colleague(
                        colleague_id, 
                        question,
                        context=f"Working on: {task}",
                    )
                    colleague_calls.append({
                        "colleague": colleague_id,
                        "question": question,
                        "result": result,
                    })
                    
                    if result.get("success"):
                        colleague_response = result.get("response", "No response")
                        if len(colleague_response) > 2000:
                            colleague_response = colleague_response[:2000] + "\n... (truncated)"
                        
                        messages.append({
                            "role": "user",
                            "content": f"COLLEAGUE RESPONSE ({result.get('colleague')}):\n{colleague_response}\n\nContinue with your task."
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"COLLEAGUE ERROR: {result.get('error')}. Available: {result.get('available_colleagues')}"
                        })
                    
                    continue
                    
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "user",
                        "content": f"ERROR: Could not parse colleague call JSON: {e}. Please use valid JSON format."
                    })
                    continue
            
            # No special blocks found - agent is just thinking/responding
            # Prompt to continue or finish
            if iterations < max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue working on the task. Use tools if needed, or signal ```done``` when complete."
                })
        
        # Max iterations reached
        if verbose:
            print(f"      [{self.display_name}] ⚠️ Max iterations reached")
        
        return {
            "success": False,
            "error": f"Max iterations ({max_iterations}) reached without completion",
            "iterations": iterations,
            "tool_calls": tool_calls,
            "colleague_calls": colleague_calls,
            "partial_result": messages[-1].get("content", "") if messages else "",
        }
    
    def _build_tool_descriptions(self) -> str:
        """Build descriptions of available tools for agentic mode."""
        if not self.tool_instances:
            return "None"
        
        descriptions = []
        for name, tool in self.tool_instances.items():
            desc = getattr(tool, 'description', 'No description')
            descriptions.append(f"- {name}: {desc}")
        
        return "\n".join(descriptions)
    
    def _build_colleague_descriptions(self) -> str:
        """Build descriptions of available colleagues."""
        if not self.colleagues:
            return "None"
        
        descriptions = []
        for agent_id, agent in self.colleagues.items():
            descriptions.append(f"- {agent_id}: {agent.display_name} ({agent.role})")
        
        return "\n".join(descriptions)

