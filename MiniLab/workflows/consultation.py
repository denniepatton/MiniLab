"""
CONSULTATION Workflow Module.

Initial phase for understanding user goals, clarifying requirements,
and establishing project scope. Led by Bohr (orchestrator).

FAST Design: Minimal LLM calls, direct data exploration.
Uses structured JSON responses for reliability.
"""

from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import csv
import json
import re

from .base import WorkflowModule, WorkflowResult, WorkflowCheckpoint, WorkflowStatus
from ..utils import console


@dataclass
class ConsultationOutput:
    """Structured output from consultation."""
    project_name: str
    goals: list[str]
    constraints: list[str]
    data_sources: list[str]
    success_criteria: list[str]
    clarifications: dict[str, str]
    recommended_workflow: str


class ConsultationModule(WorkflowModule):
    """
    CONSULTATION: The ONLY entry point for MiniLab.
    
    FAST Design:
        - Direct file scanning (no agent loops for data exploration)
        - Single LLM call for analysis + questions
        - Single user interaction
        - Single LLM call for spec creation
    
    Total: 2 LLM calls instead of 5+ agent loops
    """
    
    name = "consultation"
    description = "Initial user consultation to clarify goals and scope"
    
    required_inputs = ["user_request"]
    optional_inputs = ["existing_project", "data_paths"]
    expected_outputs = ["project_spec", "recommended_workflow", "data_manifest", "token_budget"]
    
    primary_agents = ["bohr"]
    supporting_agents = ["bayes", "gould"]  # For quick specialist consultations
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[WorkflowCheckpoint] = None,
    ) -> WorkflowResult:
        """
        Execute FAST consultation workflow.
        
        Steps:
        1. Quick direct data scan (NO agent loop)
        2. Single LLM call: analyze + generate questions
        3. User interaction
        4. Single LLM call: create spec
        """
        # Validate inputs
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        user_request = inputs["user_request"]
        
        # Initialize state
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = WorkflowStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "user_request": user_request,
                "data_manifest": {},
                "clarifications": "",
            }
        
        self._log_step(f"Starting consultation: {user_request[:80]}...")
        
        try:
            # Step 1: FAST direct data scan (no LLM call)
            if self._current_step <= 0:
                data_paths = self._extract_data_paths(user_request)
                self._state["data_manifest"] = self._quick_data_scan(data_paths)
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Quick specialist consultations for informed recommendations
            if self._current_step <= 1:
                manifest_text = self._format_manifest(self._state["data_manifest"])
                
                # Get brief statistical guidance from Bayes if data is involved
                stats_advice = ""
                if self._state["data_manifest"].get("files"):
                    bayes = self.agents.get("bayes")
                    if bayes:
                        stats_advice = await bayes.simple_query(
                            query=f"""Given this data structure, what statistical approach would you recommend? ONE paragraph max.
Data: {manifest_text}
Request: {user_request[:500]}""",
                            context="Brief consultation"
                        )
                
                self._state["stats_advice"] = stats_advice
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Bohr analyzes and proposes an optimal plan (advisory, not hand-holding)
            if self._current_step <= 2:
                bohr = self.agents.get("bohr")
                if not bohr:
                    return WorkflowResult(status=WorkflowStatus.FAILED, error="Bohr agent unavailable")
                
                manifest_text = self._format_manifest(self._state["data_manifest"])
                stats_advice = self._state.get("stats_advice", "")
                
                analysis = await bohr.simple_query(
                    query=f"""Analyze this request and propose an optimal analysis plan.

## User Request:
{user_request}

## Data Found:
{manifest_text if manifest_text else "No data files detected."}

{f"## Statistical Guidance (from Bayes):{chr(10)}{stats_advice}" if stats_advice else ""}

## Your Task:
As an expert orchestrator, propose the OPTIMAL analysis plan for this request. Be advisory and decisive - recommend what YOU think is best.

### Format your response as:

### My Understanding
[2-3 sentences summarizing what they want]

### Recommended Approach
[Your expert recommendation for how to tackle this - be specific about methods, scope, and deliverables. This is what you ADVISE, not options to choose from.]

### Recommended Budget
Based on the complexity of this request, suggest an appropriate token budget:
- Assess request complexity (narrow vs broad scope, data complexity, literature needs)
- Recommend ONE specific budget with reasoning
- Present alternatives ONLY if the user might want different scope

Example: "For this project, I recommend ~300K tokens (~$1.50). This allows for a focused literature review, thorough analysis, and complete writeup. If you want just a quick exploration, 100K would suffice."

### Key Questions (1-2 max, or none)
Ask ONLY if there's genuine ambiguity that YOU cannot resolve:
- **Primary Endpoint** - If multiple outcomes exist and it's unclear which matters
- One domain-specific question ONLY if truly essential

DO NOT ask about:
- Token budget (you recommend it)
- Timelines (we execute immediately)
- Whether they want figures (assume yes)
- Obvious scope questions (use your judgment)
- Anything you can reasonably infer or decide yourself

If the user said things like "use your best judgment" or "proceed autonomously", make ALL decisions yourself without questions.""",
                    context=f"Project: {self.project_path.name}"
                )
                
                self._state["analysis"] = analysis
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: User interaction (direct, not via agent)
            if self._current_step <= 3:
                # Pause spinner for user input
                from ..utils import Spinner
                was_spinning = Spinner.pause_for_input()
                
                console.agent_message("BOHR", self._state["analysis"])
                print()
                try:
                    user_response = input("  \033[1;32m▶ Your response:\033[0m ").strip()
                except (KeyboardInterrupt, EOFError):
                    user_response = ""
                
                # Resume spinner
                if was_spinning:
                    Spinner.resume_after_input()
                
                # Parse token budget from response (or from Bohr's recommendation if user didn't specify)
                self._state["token_budget"] = self._parse_token_budget(
                    user_response, 
                    bohr_analysis=self._state.get("analysis", "")
                )
                self._state["clarifications"] = user_response
                self._current_step = 4
                self.save_checkpoint()
            
            # Step 5: Create spec with JSON structure
            if self._current_step <= 4:
                bohr = self.agents.get("bohr")
                
                spec = await bohr.simple_query(
                    query=f"""Create a project specification based on our consultation.

## Original Request
{user_request}

## Data Available
{self._format_manifest(self._state["data_manifest"])}

## User's Clarifications
{self._state["clarifications"]}

## Output Requirements
Create a detailed project specification with clear sections.

At the END of your response, include a JSON block with workflow recommendation:
```json
{{"recommended_workflow": "WORKFLOW_NAME", "goals": ["goal1", "goal2"], "success_criteria": ["criterion1"]}}
```

Valid WORKFLOW_NAME values: brainstorming, literature_review, start_project, explore_dataset

## Create Specification:
1. **Goals** - What we're accomplishing (be specific)
2. **Data Sources** - What data we'll use and why
3. **Approach** - High-level methodology
4. **Success Criteria** - How we'll know it's done
5. **Potential Challenges** - Anticipated issues
""",
                    context=f"Project: {self.project_path.name}"
                )
                
                self._state["project_spec"] = spec
                self._current_step = 5
            
            # Extract workflow recommendation from JSON in response
            recommended = self._extract_workflow_json(self._state["project_spec"])
            
            # Write outputs
            self._write_outputs()
            
            self._status = WorkflowStatus.COMPLETED
            
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                outputs={
                    "project_spec": self._state["project_spec"],
                    "recommended_workflow": recommended,
                    "data_manifest": self._state["data_manifest"],
                    "token_budget": self._state.get("token_budget"),
                    # Pass user's natural language preferences as context for downstream agents
                    # This enables fluid, contextual autonomy - not hardcoded levels
                    "user_preferences": self._state.get("clarifications", ""),
                },
                artifacts=[str(self.project_path / "project_specification.md")],
                summary=f"Consultation complete. Recommended: {recommended}",
            )
            
        except Exception as e:
            self._status = WorkflowStatus.FAILED
            self._log_step(f"Error: {e}")
            return WorkflowResult(status=WorkflowStatus.FAILED, error=str(e))
    
    def _extract_data_paths(self, request: str) -> list[Path]:
        """Extract data paths from request."""
        paths = []
        workspace_root = self.project_path.parent.parent  # Sandbox parent
        
        for pattern in [r'ReadData/[\w/\-\.]+', r'Sandbox/[\w/\-\.]+']:
            for match in re.findall(pattern, request):
                full_path = workspace_root / match
                if full_path.exists():
                    paths.append(full_path)
        return paths
    
    def _quick_data_scan(self, paths: list[Path]) -> dict:
        """Quickly scan data files directly."""
        manifest = {"files": [], "summary": {}}
        
        for path in paths:
            items = [path] if path.is_file() else list(path.iterdir())
            for item in items:
                if item.is_file() and item.suffix in ['.csv', '.tsv', '.txt']:
                    manifest["files"].append(self._scan_file(item))
        
        manifest["summary"] = {
            "total_files": len(manifest["files"]),
            "total_rows": sum(f.get("rows", 0) for f in manifest["files"]),
        }
        return manifest
    
    def _scan_file(self, path: Path) -> dict:
        """Scan a single data file."""
        info = {"name": path.name, "path": str(path)}
        try:
            with open(path, 'r') as f:
                delimiter = '\t' if path.suffix == '.tsv' else ','
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                info["columns"] = len(header) if header else 0
                info["column_names"] = header[:15] if header else []
                info["rows"] = sum(1 for _ in reader)
        except Exception as e:
            info["error"] = str(e)
        return info
    
    def _format_manifest(self, manifest: dict) -> str:
        """Format manifest for display."""
        if not manifest.get("files"):
            return ""
        lines = [f"Found {len(manifest['files'])} file(s):"]
        for f in manifest["files"]:
            lines.append(f"• {f['name']}: {f.get('rows', '?')} rows, {f.get('columns', '?')} columns")
            if f.get("column_names"):
                cols = ", ".join(f["column_names"][:8])
                if len(f["column_names"]) > 8:
                    cols += f" (+{len(f['column_names'])-8} more)"
                lines.append(f"  Columns: {cols}")
        return "\n".join(lines)
    
    def _parse_token_budget(self, user_response: str, bohr_analysis: str = "") -> Optional[int]:
        """
        Parse token budget from user response OR Bohr's recommendation.
        
        Priority:
        1. User's explicit budget specification
        2. User's keyword (quick, thorough, comprehensive)
        3. Bohr's recommended budget from analysis
        4. Default based on implied scope
        
        Handles common typos like 'l' for 'k' (e.g., "200l" → "200k").
        """
        response_lower = user_response.lower()
        
        # Fix common typos: l→k (adjacent on keyboard)
        response_fixed = re.sub(r'(\d+)[lL]\b', r'\1k', response_lower)
        
        # Check for explicit size keywords from user
        if any(word in response_fixed for word in ['quick', 'fast', 'minimal', '100k']):
            return 100_000
        if any(word in response_fixed for word in ['thorough', 'standard', '500k']):
            return 500_000
        if any(word in response_fixed for word in ['comprehensive', 'full', 'deep', '1m', '1000k']):
            return 1_000_000
        
        # Check for explicit numbers from user (e.g., "200000 tokens" or "200k" or "200K tokens")
        number_match = re.search(r'(\d+(?:,\d+)*)\s*[kK]?\s*(?:tokens?)?', response_fixed)
        if number_match:
            num_str = number_match.group(1).replace(',', '')
            num = int(num_str)
            if 'k' in number_match.group(0).lower():
                num *= 1000
            if num >= 50_000:
                return num
            elif num > 0 and num < 50_000:
                console.warning(f"Budget '{num}' seems low. Interpreting as {num}k = {num * 1000:,} tokens.")
                return num * 1000
        
        # If user didn't specify, try to extract Bohr's recommendation
        if bohr_analysis:
            bohr_lower = bohr_analysis.lower()
            # Look for Bohr's specific recommendation (e.g., "~300K", "recommend 500k")
            bohr_match = re.search(r'recommend[^\d]*(\d+)\s*[kK]', bohr_lower)
            if bohr_match:
                bohr_budget = int(bohr_match.group(1)) * 1000
                console.info(f"Using Bohr's recommended budget: {bohr_budget:,} tokens")
                return bohr_budget
            
            # Look for any specific K value in recommendation section
            bohr_match = re.search(r'~?(\d+)\s*[kK]\s*tokens?', bohr_lower)
            if bohr_match:
                bohr_budget = int(bohr_match.group(1)) * 1000
                if bohr_budget >= 50_000:
                    console.info(f"Using Bohr's suggested budget: {bohr_budget:,} tokens")
                    return bohr_budget
        
        # Default based on user's autonomy signals
        if any(word in response_lower for word in ['best judgment', 'autonomous', 'your discretion', 'proceed']):
            # User wants us to decide - use moderate budget
            console.info("User trusts our judgment. Using standard budget: 300K tokens.")
            return 300_000
        
        # Final default
        console.info("No budget specified. Using standard budget: 300K tokens.")
        return 300_000
    
    def _extract_workflow_json(self, spec: str) -> str:
        """
        Extract workflow recommendation from JSON block in spec.
        
        Looks for ```json block with recommended_workflow field.
        Falls back to keyword detection if JSON not found.
        """
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', spec, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                workflow = data.get("recommended_workflow", "").lower()
                if workflow in ["brainstorming", "literature_review", "start_project", "explore_dataset"]:
                    return workflow
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for workflow mentions in text
        spec_lower = spec.lower()
        if "explore" in spec_lower and "data" in spec_lower:
            return "explore_dataset"
        if "literature" in spec_lower or "review" in spec_lower:
            return "literature_review"
        if "brainstorm" in spec_lower:
            return "brainstorming"
        
        return "start_project"
    
    def _write_outputs(self) -> None:
        """Write consultation outputs."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Write project specification
        with open(self.project_path / "project_specification.md", "w") as f:
            f.write("# Project Specification\n\n")
            f.write(self._state.get("project_spec", ""))
        
        # Only write data manifest if there are actual data files
        data_manifest = self._state.get("data_manifest", {})
        if data_manifest.get("files"):
            with open(self.project_path / "data_manifest.md", "w") as f:
                f.write("# Data Manifest\n\n")
                f.write(self._format_manifest(data_manifest))
