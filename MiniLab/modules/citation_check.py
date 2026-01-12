"""
CITATION_CHECK Module.

Linear module for verifying citations and references.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class CitationCheckModule(Module):
    """
    CITATION_CHECK: Verify citations and references.
    
    Linear flow:
    1. Parse document for citations
    2. Check each citation exists in bibliography
    3. Verify citation format consistency
    4. Flag missing or malformed citations
    5. Check for uncited references
    
    Primary Agent: Gould
    
    Outputs:
        - Citation audit report
        - Missing citations
        - Format issues
        - Uncited references
    """
    
    name = "citation_check"
    description = "Verify citations and references"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["document_path"]
    optional_inputs = ["bibliography_path", "citation_style", "strict_mode"]
    expected_outputs = ["passed", "missing_citations", "format_issues", "uncited_refs"]
    
    primary_agents = ["gould"]
    supporting_agents = []
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute citation check."""
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
                "bibliography_path": inputs.get("bibliography_path", ""),
                "citation_style": inputs.get("citation_style", "author-year"),
                "strict_mode": inputs.get("strict_mode", False),
            }
        
        self._log_step(f"Checking citations in: {document_path.name}")
        
        try:
            # Read document
            with open(document_path, "r") as f:
                document_content = f.read()
            
            # Step 1: Parse citations
            if self._current_step <= 0:
                self._log_step("Step 1: Parsing citations from document")
                
                parse_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Parse citations from this document.

Document excerpt:
{document_content[:4000]}

Citation style expected: {self._state['citation_style']}

Extract all citations and list them:
1. In-text citation as it appears
2. Inferred reference (author, year)
3. Location in document (approximate)

Also note any citation format inconsistencies.""",
                )
                
                self._state["parsed_citations"] = parse_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Check against bibliography
            if self._current_step <= 1:
                self._log_step("Step 2: Checking against bibliography")
                
                bib_content = ""
                if self._state.get("bibliography_path"):
                    bib_path = Path(self._state["bibliography_path"])
                    if bib_path.exists():
                        with open(bib_path, "r") as f:
                            bib_content = f.read()
                
                # Also check memory/sources/ if no explicit bibliography
                if not bib_content:
                    sources_dir = self.project_path / "memory" / "sources"
                    if sources_dir.exists():
                        bib_files = list(sources_dir.glob("*.md")) + list(sources_dir.glob("*.bib"))
                        for bf in bib_files[:5]:  # Limit to avoid too much content
                            with open(bf, "r") as f:
                                bib_content += f"\n\n--- {bf.name} ---\n" + f.read()
                
                check_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Verify citations against bibliography.

Parsed Citations:
{self._state['parsed_citations']}

Bibliography:
{bib_content[:3000] if bib_content else 'No bibliography provided'}

For each citation:
1. Can it be matched to a bibliography entry?
2. Is the citation format correct?
3. Any discrepancies (year, spelling)?

List:
- MISSING: Citations with no bibliography match
- FORMAT ISSUES: Citations with format problems
- MATCHES: Successfully verified citations""",
                )
                
                self._state["verification_result"] = check_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Check for uncited references
            if self._current_step <= 2:
                self._log_step("Step 3: Checking for uncited references")
                
                uncited_result = await self._run_agent_task(
                    agent_name="gould",
                    task=f"""Check for uncited references in bibliography.

Parsed Citations:
{self._state['parsed_citations']}

Verification Result:
{self._state['verification_result']}

Identify any references in the bibliography that are NOT cited in the document.
List them as UNCITED REFERENCES.

Also provide:
1. Total citations in document
2. Total unique references
3. Coverage assessment""",
                )
                
                self._state["uncited_refs"] = uncited_result.get("response", "")
                self._current_step = 3
            
            # Determine pass/fail
            verification = self._state.get("verification_result", "").upper()
            has_missing = "MISSING:" in verification and "MISSING: NONE" not in verification.upper()
            has_format_issues = "FORMAT ISSUES:" in verification and "FORMAT ISSUES: NONE" not in verification.upper()
            
            strict = self._state.get("strict_mode", False)
            passed = not has_missing and (not strict or not has_format_issues)
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "passed": passed,
                    "missing_citations": has_missing,
                    "format_issues": has_format_issues,
                    "uncited_refs": self._state.get("uncited_refs", ""),
                    "verification_report": self._state.get("verification_result", ""),
                },
                summary=f"Citation check {'passed' if passed else 'found issues'}.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
