from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from MiniLab.llm_backends.base import ChatMessage, LLMBackend


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

