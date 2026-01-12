"""
SANITY_CHECK_DATA Module.

Linear module for rapid data sanity checking before analysis.
"""

from typing import Any, Optional
from pathlib import Path
import json

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class SanityCheckDataModule(Module):
    """
    SANITY_CHECK_DATA: Quick data integrity checks.
    
    Linear flow:
    1. Check file existence and format
    2. Validate dimensions and types
    3. Check for missing values, outliers
    4. Verify expected columns/features
    5. Report issues
    
    Primary Agent: Dayhoff
    
    Outputs:
        - Sanity check report
        - Pass/fail status
        - Recommended actions
    """
    
    name = "sanity_check_data"
    description = "Rapid data sanity checking before analysis"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["data_path"]
    optional_inputs = ["expected_columns", "expected_rows_min", "expected_rows_max", "data_type"]
    expected_outputs = ["passed", "issues", "recommendations", "summary"]
    
    primary_agents = ["dayhoff"]
    supporting_agents = ["bayes"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute data sanity check."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        data_path = Path(inputs["data_path"])
        if not data_path.exists():
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Data file not found: {data_path}",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "data_path": str(data_path),
                "expected_columns": inputs.get("expected_columns", []),
                "expected_rows_min": inputs.get("expected_rows_min"),
                "expected_rows_max": inputs.get("expected_rows_max"),
                "data_type": inputs.get("data_type", "csv"),
                "issues": [],
            }
        
        self._log_step(f"Sanity checking: {data_path.name}")
        
        try:
            # Step 1: Check file and load
            if self._current_step <= 0:
                self._log_step("Step 1: Checking file format and loading")
                
                check_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Perform initial data sanity checks.

Data file: {inputs['data_path']}
Expected type: {self._state['data_type']}

Check:
1. File exists and is readable
2. File format matches expected type
3. Load the data and report basic info:
   - Number of rows
   - Number of columns
   - Column names
   - Data types
4. Note any loading errors or warnings

Use pandas or appropriate library to load and inspect.""",
                )
                
                self._state["initial_check"] = check_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
            
            # Step 2: Validate structure
            if self._current_step <= 1:
                self._log_step("Step 2: Validating data structure")
                
                expected_cols = self._state.get("expected_columns", [])
                cols_note = f"Expected columns: {expected_cols}" if expected_cols else "No specific columns expected"
                
                row_constraints = []
                if self._state.get("expected_rows_min"):
                    row_constraints.append(f"Min rows: {self._state['expected_rows_min']}")
                if self._state.get("expected_rows_max"):
                    row_constraints.append(f"Max rows: {self._state['expected_rows_max']}")
                rows_note = ", ".join(row_constraints) if row_constraints else "No row count constraints"
                
                struct_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Validate data structure against expectations.

Initial Check:
{self._state['initial_check'][:2000]}

{cols_note}
{rows_note}

Validate:
1. All expected columns present
2. Row count within range
3. Column data types appropriate
4. Index/ID columns unique

List any structural issues found.""",
                )
                
                self._state["structure_check"] = struct_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Check data quality
            if self._current_step <= 2:
                self._log_step("Step 3: Checking data quality")
                
                quality_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Check data quality for {inputs['data_path']}.

Check for:
1. Missing values (count per column, % missing)
2. Duplicate rows
3. Outliers (for numeric columns)
4. Invalid values (negative where shouldn't be, etc.)
5. Encoding issues

Report issues with severity:
- CRITICAL: Data unusable without fixing
- WARNING: Should address but can proceed
- INFO: Minor issues to note

Provide summary statistics for key columns.""",
                )
                
                self._state["quality_check"] = quality_result.get("response", "")
                self._current_step = 3
                self.save_checkpoint()
            
            # Step 4: Generate recommendations
            if self._current_step <= 3:
                self._log_step("Step 4: Generating recommendations")
                
                rec_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Based on the sanity checks, provide recommendations.

Initial Check:
{self._state['initial_check'][:1000]}

Structure Check:
{self._state['structure_check'][:1000]}

Quality Check:
{self._state['quality_check'][:1000]}

Provide:
1. OVERALL STATUS: PASS / PASS_WITH_WARNINGS / FAIL
2. Critical issues that must be fixed
3. Recommended preprocessing steps
4. Data transformation suggestions
5. Whether analysis can proceed

Be specific about what needs to be done.""",
                )
                
                self._state["recommendations"] = rec_result.get("response", "")
                self._current_step = 4
            
            # Determine pass/fail
            response = self._state.get("recommendations", "").upper()
            passed = "PASS" in response and "FAIL" not in response
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "passed": passed,
                    "issues": self._state.get("quality_check", ""),
                    "recommendations": self._state.get("recommendations", ""),
                    "summary": f"Data sanity check {'PASSED' if passed else 'FAILED'} for {data_path.name}",
                },
                summary=f"Data sanity check {'passed' if passed else 'failed'}.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
