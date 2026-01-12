"""
FORMATTING_CHECK Module.

Linear module for automated rubric checking of documents.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class FormattingCheckModule(Module):
    """
    FORMATTING_CHECK: Automated rubric checking.
    
    Linear flow:
    1. Load formatting rubric
    2. Parse document structure
    3. Check against each rubric criterion
    4. Generate pass/fail report
    5. Suggest fixes for failures
    
    Primary Agent: Farber
    
    Outputs:
        - Rubric compliance report
        - Pass/fail per criterion
        - Suggested fixes
    """
    
    name = "formatting_check"
    description = "Automated rubric checking for documents"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["document_path"]
    optional_inputs = ["rubric_path", "rubric_type", "auto_fix"]
    expected_outputs = ["passed", "compliance_report", "suggested_fixes"]
    
    primary_agents = ["farber"]
    supporting_agents = []
    
    # Default rubric criteria (can be overridden)
    DEFAULT_RUBRIC = {
        "heading_hierarchy": "Headings follow proper hierarchy (# > ## > ###)",
        "no_orphan_headings": "No single subheading under a parent",
        "consistent_lists": "List formatting is consistent (all bullets or all numbers)",
        "code_block_language": "Code blocks specify language",
        "table_headers": "Tables have header rows",
        "link_validity": "Links use proper markdown format",
        "no_raw_urls": "URLs are wrapped in markdown links",
        "figure_captions": "Figures have captions",
        "section_completeness": "Required sections are present",
        "consistent_terminology": "Terminology is used consistently",
    }
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute formatting check."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        document_path = Path(inputs["document_path"])
        if not document_path.exists():
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Document not found: {document_path}",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "document_path": str(document_path),
                "rubric_path": inputs.get("rubric_path", ""),
                "rubric_type": inputs.get("rubric_type", "default"),
                "auto_fix": inputs.get("auto_fix", False),
                "criteria_results": {},
            }
        
        self._log_step(f"Checking formatting: {document_path.name}")
        
        try:
            # Read document
            with open(document_path, "r") as f:
                document_content = f.read()
            
            # Load rubric
            rubric = self._load_rubric()
            
            # Step 1: Parse document structure
            if self._current_step <= 0:
                self._log_step("Step 1: Parsing document structure")
                
                parse_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Analyze the structure of this document.

Document:
{document_content[:4000]}

Identify:
1. Heading structure (levels, hierarchy)
2. List types used
3. Code blocks (with/without language)
4. Tables (with/without headers)
5. Links and URLs
6. Figures and captions
7. Section organization

Provide a structural summary.""",
                )
                
                self._state["document_structure"] = parse_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Check each criterion
            if self._current_step <= 1:
                self._log_step("Step 2: Checking against rubric")
                
                rubric_text = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])
                
                check_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Check document against formatting rubric.

Document Structure:
{self._state['document_structure']}

Document Content:
{document_content[:3000]}

Rubric Criteria:
{rubric_text}

For each criterion:
1. PASS or FAIL
2. Evidence (specific examples)
3. Location of issues (if FAIL)

Be specific about failures.""",
                )
                
                self._state["rubric_check"] = check_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Generate suggested fixes
            if self._current_step <= 2:
                self._log_step("Step 3: Generating fix suggestions")
                
                fix_result = await self._run_agent_task(
                    agent_name="farber",
                    task=f"""Generate specific fixes for formatting issues.

Rubric Check Results:
{self._state['rubric_check']}

For each FAIL:
1. What exactly needs to change
2. How to fix it (specific edit)
3. Example of correct format

Prioritize fixes by importance.
Also provide an OVERALL score (X/Y criteria passed).""",
                )
                
                self._state["suggested_fixes"] = fix_result.get("response", "")
                self._current_step = 3
            
            # Determine pass/fail
            check_result = self._state.get("rubric_check", "").upper()
            fail_count = check_result.count("FAIL")
            pass_count = check_result.count("PASS")
            total = fail_count + pass_count
            passed = fail_count == 0 or (fail_count / total < 0.3 if total > 0 else True)
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "passed": passed,
                    "compliance_report": self._state.get("rubric_check", ""),
                    "suggested_fixes": self._state.get("suggested_fixes", ""),
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                },
                summary=f"Formatting check: {pass_count}/{total} criteria passed.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _load_rubric(self) -> dict[str, str]:
        """Load formatting rubric."""
        rubric = self.DEFAULT_RUBRIC.copy()
        
        # Try to load custom rubric
        rubric_path = self._state.get("rubric_path", "")
        if rubric_path:
            path = Path(rubric_path)
            if path.exists():
                try:
                    import yaml
                    with open(path, "r") as f:
                        custom = yaml.safe_load(f)
                    if isinstance(custom, dict):
                        rubric.update(custom)
                except Exception:
                    pass
        
        # Also check config/formatting_rubric.md
        config_rubric = Path(__file__).parent.parent / "config" / "formatting_rubric.md"
        if config_rubric.exists():
            # Parse markdown rubric (simplified)
            pass
        
        return rubric
