"""
GENERATE_CODE Module.

Linear module for writing scripts and code artifacts.
Follows CellVoyager pattern: Dayhoff (structure) → Hinton (implement) → Bayes (verify).
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class GenerateCodeModule(Module):
    """
    GENERATE_CODE: Write scripts to scripts/, optionally run.
    
    CellVoyager pattern:
    1. Dayhoff: Define structure, inputs/outputs, skeleton
    2. Hinton: Implement full code
    3. Bayes: Review for correctness, statistics issues
    4. (Optional) Run the code
    
    Primary Agent: Dayhoff → Hinton → Bayes
    
    Outputs:
        - scripts/{script_name}.py
        - Execution output (if run=True)
    """
    
    name = "generate_code"
    description = "Write scripts using CellVoyager pattern"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["script_name", "purpose", "specifications"]
    optional_inputs = ["input_data", "output_format", "run_after_generation", "existing_code"]
    expected_outputs = ["script_path", "code_content", "execution_output"]
    
    primary_agents = ["dayhoff", "hinton", "bayes"]
    supporting_agents = []
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute code generation."""
        valid, missing = self.validate_inputs(inputs)
        if not valid:
            return ModuleResult(
                status=ModuleStatus.FAILED,
                error=f"Missing required inputs: {missing}",
            )
        
        if checkpoint:
            self.restore(checkpoint)
        else:
            self._status = ModuleStatus.IN_PROGRESS
            self._current_step = 0
            self._state = {
                "script_name": inputs["script_name"],
                "purpose": inputs["purpose"],
                "specifications": inputs["specifications"],
                "input_data": inputs.get("input_data", ""),
                "output_format": inputs.get("output_format", ""),
                "run_after": inputs.get("run_after_generation", False),
                "existing_code": inputs.get("existing_code", ""),
            }
        
        self._log_step(f"Generating code: {inputs['script_name']}")
        
        try:
            # Step 1: Dayhoff structures the code
            if self._current_step <= 0:
                self._log_step("Step 1: Dayhoff structures the code")
                
                existing = self._state.get("existing_code", "")
                existing_note = f"\n\nExisting code to modify:\n```python\n{existing[:2000]}\n```" if existing else ""
                
                structure_result = await self._run_agent_task(
                    agent_name="dayhoff",
                    task=f"""Design the code structure for: {inputs['script_name']}

Purpose: {inputs['purpose']}

Specifications:
{inputs['specifications']}

Input Data: {self._state.get('input_data', 'Not specified')}
Output Format: {self._state.get('output_format', 'Not specified')}
{existing_note}

Create a detailed code skeleton including:
1. Required imports
2. Function signatures with docstrings
3. Main execution flow
4. Input/output handling
5. Error handling approach

Focus on biological/scientific data best practices.""",
                )
                
                self._state["structure"] = structure_result.get("response", "")
                self._current_step = 1
                self.save_checkpoint()
                console.agent_message("DAYHOFF", "Code structure designed")
            
            # Step 2: Hinton implements
            if self._current_step <= 1:
                self._log_step("Step 2: Hinton implements the code")
                
                impl_result = await self._run_agent_task(
                    agent_name="hinton",
                    task=f"""Implement the full code based on this structure.

Structure and skeleton:
{self._state['structure']}

Purpose: {inputs['purpose']}
Input Data: {self._state.get('input_data', 'Not specified')}
Output Format: {self._state.get('output_format', 'Not specified')}

Implement complete, working code that:
1. Follows the structure exactly
2. Uses appropriate libraries (pandas, numpy, sklearn, etc.)
3. Includes proper error handling
4. Has informative logging/output
5. Is well-commented

Return ONLY the Python code in a code block.""",
                )
                
                # Extract code from response
                response = impl_result.get("response", "")
                import re
                code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    self._state["implementation"] = code_match.group(1)
                else:
                    # Try to extract any code block
                    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                    if code_match:
                        self._state["implementation"] = code_match.group(1)
                    else:
                        self._state["implementation"] = response
                
                self._current_step = 2
                self.save_checkpoint()
                console.agent_message("HINTON", "Code implemented")
            
            # Step 3: Bayes reviews
            if self._current_step <= 2:
                self._log_step("Step 3: Bayes reviews the code")
                
                review_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Review this code for correctness, especially statistical issues.

Code:
```python
{self._state['implementation'][:4000]}
```

Purpose: {inputs['purpose']}
Specifications: {inputs['specifications'][:500]}

Review for:
1. Statistical correctness (appropriate tests, assumptions)
2. Numerical stability
3. Data handling issues
4. Logic errors
5. Edge cases

If changes needed, return the corrected code in a code block.
If code is good, say "APPROVED" and explain why.""",
                )
                
                response = review_result.get("response", "")
                
                # Check if approved or needs changes
                if "APPROVED" in response.upper():
                    self._state["final_code"] = self._state["implementation"]
                else:
                    # Extract corrected code
                    import re
                    code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
                    if code_match:
                        self._state["final_code"] = code_match.group(1)
                    else:
                        self._state["final_code"] = self._state["implementation"]
                
                self._state["review_notes"] = response
                self._current_step = 3
                console.agent_message("BAYES", "Code reviewed")
            
            # Write the code
            script_path = self._write_code()
            
            # Step 4: Optionally run the code
            execution_output = ""
            if self._state.get("run_after", False):
                self._log_step("Step 4: Running the code")
                # This would integrate with terminal tool
                # For now, just note it
                execution_output = "Code written. Manual execution required."
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "script_path": script_path,
                    "code_content": self._state.get("final_code", ""),
                    "execution_output": execution_output,
                    "review_notes": self._state.get("review_notes", ""),
                },
                artifacts=[script_path],
                summary=f"Script {inputs['script_name']} generated and reviewed.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_code(self) -> str:
        """Write code to scripts/ directory."""
        scripts_dir = self.project_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_name = self._state["script_name"]
        if not script_name.endswith(".py"):
            script_name += ".py"
        
        filepath = scripts_dir / script_name
        
        # Add header
        header = f'''"""
{self._state['purpose']}

Generated by MiniLab GENERATE_CODE module.
Reviewed by Bayes.
"""

'''
        
        code = self._state.get("final_code", "")
        if not code.startswith('"""') and not code.startswith("'''"):
            code = header + code
        
        with open(filepath, "w") as f:
            f.write(code)
        
        return str(filepath)
