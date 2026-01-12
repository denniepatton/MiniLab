"""
WRITE_ARTIFACT Module.

Linear module for creating or updating structured documents in artifacts/.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class WriteArtifactModule(Module):
    """
    WRITE_ARTIFACT: Create or update structured documents.
    
    Linear flow:
    1. Determine artifact type and structure
    2. Generate content based on inputs
    3. Apply formatting rubric
    4. Write to artifacts/
    
    Primary Agent: Farber
    
    Outputs:
        - New or updated artifact file
    """
    
    name = "write_artifact"
    description = "Create or update structured documents in artifacts/"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["artifact_name", "content_type", "content_inputs"]
    optional_inputs = ["existing_content", "formatting_requirements", "sections"]
    expected_outputs = ["artifact_path", "artifact_content"]
    
    primary_agents = ["farber"]
    supporting_agents = ["bohr"]
    
    ARTIFACT_TYPES = {
        "plan": {
            "filename": "plan.md",
            "sections": ["Summary", "Phases", "Delegation", "Budget", "Outputs"],
        },
        "evidence": {
            "filename": "evidence.md",
            "sections": ["Evidence Packets", "Sources", "Quality Assessment"],
        },
        "decisions": {
            "filename": "decisions.md",
            "sections": ["Key Decisions", "Rationale", "Tradeoffs"],
        },
        "interpretation": {
            "filename": "interpretation.md",
            "sections": ["Key Findings", "Statistical Summary", "Implications"],
        },
        "custom": {
            "filename": None,  # Must be specified
            "sections": [],  # Must be specified
        },
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute artifact writing."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        content_type = inputs["content_type"].lower()
        artifact_name = inputs["artifact_name"]
        
        if content_type not in self.ARTIFACT_TYPES:
            content_type = "custom"
        
        type_config = self.ARTIFACT_TYPES[content_type]
        filename = type_config["filename"] or f"{artifact_name}.md"
        sections = inputs.get("sections", type_config["sections"])
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "artifact_name": artifact_name,
                "content_type": content_type,
                "filename": filename,
                "sections": sections,
                "content_inputs": inputs["content_inputs"],
                "existing_content": inputs.get("existing_content", ""),
            }
        
        self._log_step(f"Writing artifact: {filename}")
        
        try:
            # Step 1: Generate content
            if self._current_step <= 0:
                self._log_step("Step 1: Generating artifact content")
                
                sections_text = "\n".join(f"- {s}" for s in sections) if sections else "Structure as appropriate"
                
                existing = self._state.get("existing_content", "")
                existing_note = f"\n\nExisting content to update:\n{existing[:2000]}" if existing else ""
                
                content_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Create/update artifact: {artifact_name}

Content Type: {content_type}
Required Sections:
{sections_text}

Content Inputs:
{str(inputs['content_inputs'])[:3000]}
{existing_note}

Write a well-structured markdown document.
Use clear headings, bullet points, and tables where appropriate.
Be concise but complete.

If updating existing content, preserve structure where possible.""",
                )
                
                self._state["generated_content"] = content_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Apply formatting checks
            if self._current_step <= 1:
                self._log_step("Step 2: Applying formatting standards")
                
                # Quick formatting pass
                format_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Review and finalize this artifact for formatting.

Document:
{self._state['generated_content'][:4000]}

Ensure:
1. Consistent heading hierarchy (# > ## > ###)
2. Proper markdown formatting
3. No broken links or references
4. Tables are properly formatted
5. Code blocks use correct syntax highlighting

Return the finalized document.""",
                )
                
                self._state["final_content"] = format_result.get("response", "")
                self._current_step = 2
            
            # Write the artifact
            self._write_artifact()
            
            self._status = ModuleStatus.COMPLETED
            artifact_path = str(self.project_path / "artifacts" / self._state["filename"])
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "artifact_path": artifact_path,
                    "artifact_content": self._state.get("final_content", ""),
                },
                artifacts=[artifact_path],
                summary=f"Artifact {filename} written successfully.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_artifact(self) -> None:
        """Write artifact to disk."""
        artifacts_dir = self.project_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = artifacts_dir / self._state["filename"]
        content = self._state.get("final_content", self._state.get("generated_content", ""))
        
        with open(filepath, "w") as f:
            f.write(content)
