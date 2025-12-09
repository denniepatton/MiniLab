from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from MiniLab.llm_backends.base import ChatMessage, LLMBackend

if TYPE_CHECKING:
    from MiniLab.storage.transcript import TranscriptLogger


# Maximum iterations for agentic loops to prevent runaway
MAX_AGENTIC_ITERATIONS = 30  # Increased for ReAct-style small iterations


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
        - Clarification on a suggestion from another, particular agent
        - Quick statistical/methodological advice
        - Domain expertise outside your explicit specialty
        - Verification of the validity of an approach before implementing
        
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
                    "- Do not reference timelines like 'over the coming days/weeks' - everything happens now\n"
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
        logger: Optional['TranscriptLogger'] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with full agentic capabilities (ReAct-style loop).
        
        The agent can:
        - Think about what to do next
        - Use tools and see results
        - Consult colleagues and incorporate their input
        - Iterate until the task is complete
        
        Args:
            task: The task to accomplish
            context: Additional context
            max_iterations: Safety limit on iterations
            max_tokens_per_step: Tokens per reasoning step
            verbose: Print progress
            logger: Optional TranscriptLogger for logging all operations
            
        Returns:
            Dict with 'success', 'result', 'iterations', 'tool_calls', etc.
        """
        # Build tool descriptions
        tool_descriptions = self._build_tool_descriptions()
        colleague_descriptions = self._build_colleague_descriptions()
        
        # Get workspace root for path guidance
        import os
        workspace_root = os.getcwd()
        
        system_prompt = f"""You are {self.display_name}, {self.role} in the MiniLab team.

{self.persona}

You are operating in AGENTIC MODE. You can:
1. THINK about what to do
2. USE TOOLS to take actions  
3. ASK COLLEAGUES for help
4. OBSERVE results and continue

WORKSPACE ENVIRONMENT:
- Working directory: {workspace_root}
- Data location: ReadData/ (READ-ONLY - explore but don't modify)
- Output location: Sandbox/ (READ-WRITE - create scripts, outputs here)
- Python environment: micromamba run -n minilab python <script.py>
- All paths are RELATIVE to workspace root (e.g., "Sandbox/ProjectName/scripts/")

CRITICAL WORKFLOW FOR SCRIPTS:
1. FIRST create the script file using code_editor.create
2. THEN run it using terminal.execute with: micromamba run -n minilab python Sandbox/path/to/script.py
3. If errors occur, use code_editor.edit to fix, then run again
4. NEVER try to run a script before creating it!

OPERATIONAL CONSTRAINTS:
- You have {max_iterations} iterations maximum for this task
- Each iteration = one response from you (tool call or thinking)
- Work efficiently: use tools to gather info, then produce output
- If running low on iterations and need more time, you can REQUEST MORE:
  ```extend
  {{"request": "need_more_iterations", "current_progress": "what you've done so far", "remaining_work": "what still needs to be done", "additional_needed": 10}}
  ```
- If at iteration {max_iterations - 2} or later: either finish OR request extension

AVAILABLE TOOLS:
{tool_descriptions}

AVAILABLE COLLEAGUES:
{colleague_descriptions}

HOW TO USE TOOLS:
When you want to use a tool, output a JSON block like this:
```tool
{{"tool": "tool_name", "action": "action_name", "params": {{"key": "value"}}}}
```

COMMON TOOL PATTERNS:
- Create a file: {{"tool": "code_editor", "action": "create", "params": {{"path": "Sandbox/Project/scripts/my_script.py", "content": "import pandas..."}}}}
- Run a script: {{"tool": "terminal", "action": "execute", "params": {{"command": "micromamba run -n minilab python Sandbox/Project/scripts/my_script.py"}}}}
- List files: {{"tool": "terminal", "action": "execute", "params": {{"command": "ls -la ReadData/"}}}}
- Read file: {{"tool": "filesystem", "action": "read", "params": {{"path": "ReadData/file.csv"}}}}

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
2. Use tools/colleagues as needed (but don't overdo it!)
3. Observe results and adjust
4. When complete OR running low on iterations, signal done with what you have

Be concise. Act, don't just describe what you would do. Produce outputs, not plans.
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
        current_action = "thinking"
        
        # Helper for single-line status updates
        def _update_status(msg: str, final: bool = False):
            """Update the status line (overwrites previous on same line)."""
            if verbose:
                end = "\n" if final else "\r"
                # Clear line and print new status
                print(f"      [{self.display_name}] {msg}".ljust(80), end=end, flush=True)
        
        while iterations < max_iterations:
            iterations += 1
            _update_status(f"Iteration {iterations}/{max_iterations}: {current_action}...")
            
            # Get agent's response
            response = await self.backend.acomplete(messages, max_tokens=max_tokens_per_step)
            
            # Add response to conversation
            messages.append({"role": "assistant", "content": response})
            
            # Check for extension request
            extend_match = re.search(r'```extend\s*\n(.*?)\n```', response, re.DOTALL)
            if extend_match:
                try:
                    extend_data = json.loads(extend_match.group(1))
                    additional = min(extend_data.get("additional_needed", 10), 20)  # Cap at 20
                    
                    max_iterations += additional
                    _update_status(f"Extended by {additional} iterations (now {max_iterations} max)", final=True)
                    
                    messages.append({
                        "role": "user",
                        "content": f"EXTENSION GRANTED: You now have {max_iterations - iterations} more iterations. Continue working."
                    })
                    continue
                except json.JSONDecodeError:
                    pass
            
            # Check for completion
            done_match = re.search(r'```done\s*\n(.*?)\n```', response, re.DOTALL)
            if done_match:
                try:
                    done_data = json.loads(done_match.group(1))
                    _update_status(f"✓ Complete after {iterations} iterations", final=True)
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
                    
                    current_action = f"{tool_name}.{action}"
                    _update_status(f"Iteration {iterations}/{max_iterations}: 🔧 {current_action}")
                    
                    # Execute the tool
                    result = await self.use_tool(tool_name, action, **params)
                    tool_calls.append({
                        "tool": tool_name,
                        "action": action,
                        "params": params,
                        "result": result,
                    })
                    
                    # Log the tool operation
                    if logger:
                        logger.log_tool_operation(
                            self.display_name,
                            tool_name,
                            action,
                            params,
                            result
                        )
                    
                    # Format result for agent
                    result_str = json.dumps(result, indent=2, default=str)
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + "\n... (truncated)"
                    
                    # Add iteration awareness to help agent pace itself
                    remaining = max_iterations - iterations
                    urgency = ""
                    if remaining <= 2:
                        urgency = f"\n⚠️ LOW ITERATIONS: {remaining} left. Wrap up and signal done soon!"
                    elif remaining <= 4:
                        urgency = f"\n📊 Iterations remaining: {remaining}. Start producing final output."
                    
                    # Add observation to conversation
                    messages.append({
                        "role": "user",
                        "content": f"TOOL RESULT ({tool_name}.{action}):\n```\n{result_str}\n```{urgency}\n\nContinue with your task."
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
                    
                    current_action = f"asking {colleague_id}"
                    _update_status(f"Iteration {iterations}/{max_iterations}: 💬 {current_action}")
                    
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
                        
                        # Log the consultation
                        if logger:
                            logger.log_agent_consultation(
                                from_agent=self.display_name,
                                to_agent=result.get("colleague", colleague_id),
                                question=question,
                                response=colleague_response[:500],  # Truncate for log
                                tokens_used=500,  # Estimate
                            )
                        
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
            current_action = "thinking"
            # Prompt to continue or finish
            if iterations < max_iterations - 1:
                messages.append({
                    "role": "user",
                    "content": "Continue working on the task. Use tools if needed, or signal ```done``` when complete."
                })
        
        # Max iterations reached
        _update_status(f"⚠️ Max iterations ({max_iterations}) reached", final=True)
        
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

