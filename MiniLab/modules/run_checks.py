"""
RUN_CHECKS Module.

Linear module for running terminal commands and validating output.
"""

from typing import Any, Optional
from pathlib import Path

from .base import Module, ModuleResult, ModuleCheckpoint, ModuleStatus, ModuleType
from ..utils import console


class RunChecksModule(Module):
    """
    RUN_CHECKS: Execute terminal command, capture & validate output.
    
    Linear flow:
    1. Validate command safety (PathGuard)
    2. Execute command
    3. Capture output
    4. Validate output against expectations
    5. Report success/failure
    
    Primary Agent: Hinton
    
    Outputs:
        - Command output
        - Validation result
        - logs/{check_name}.log
    """
    
    name = "run_checks"
    description = "Execute terminal commands and validate output"
    module_type = ModuleType.LINEAR
    
    required_inputs = ["check_name", "command"]
    optional_inputs = ["expected_output", "validation_criteria", "timeout", "cwd"]
    expected_outputs = ["output", "exit_code", "validation_passed", "log_path"]
    
    primary_agents = ["hinton"]
    supporting_agents = ["bayes"]
    
    async def execute(
        self,
        inputs: dict[str, Any],
        checkpoint: Optional[ModuleCheckpoint] = None,
    ) -> ModuleResult:
        """Execute run checks."""
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
                "check_name": inputs["check_name"],
                "command": inputs["command"],
                "expected_output": inputs.get("expected_output", ""),
                "validation_criteria": inputs.get("validation_criteria", ""),
                "timeout": inputs.get("timeout", 300),
                "cwd": inputs.get("cwd", str(self.project_path)),
            }
        
        self._log_step(f"Running check: {inputs['check_name']}")
        
        try:
            # Step 1: Validate command safety
            if self._current_step <= 0:
                self._log_step("Step 1: Validating command safety")
                
                # Basic safety checks
                command = inputs["command"]
                dangerous_patterns = ["rm -rf /", ":(){ :|:& };:", "dd if=", "> /dev/sd"]
                for pattern in dangerous_patterns:
                    if pattern in command:
                        return ModuleResult(
                            status=ModuleStatus.FAILED,
                            error=f"Potentially dangerous command pattern detected: {pattern}",
                        )
                
                self._current_step = 1
            
            # Step 2: Execute command
            if self._current_step <= 1:
                self._log_step("Step 2: Executing command")
                
                # Use agent to run the command (integrates with terminal tool)
                exec_result = await self._run_agent_task(
                    agent_name="hinton",
                    task=f"""Execute this command and capture the output.

Command: {inputs['command']}
Working Directory: {self._state['cwd']}
Timeout: {self._state['timeout']} seconds

Run the command and report:
1. The exact command executed
2. The complete stdout
3. The complete stderr
4. The exit code
5. Any errors encountered

If the command requires input, note that.""",
                )
                
                self._state["execution_result"] = exec_result.get("response", "")
                self._current_step = 2
                self.save_checkpoint()
            
            # Step 3: Validate output
            validation_passed = True
            if self._current_step <= 2 and (self._state.get("expected_output") or self._state.get("validation_criteria")):
                self._log_step("Step 3: Validating output")
                
                validation_result = await self._run_agent_task(
                    agent_name="bayes",
                    task=f"""Validate the command output.

Command: {inputs['command']}
Output:
{self._state['execution_result'][:3000]}

Expected Output: {self._state.get('expected_output', 'Not specified')}
Validation Criteria: {self._state.get('validation_criteria', 'Command should complete successfully')}

Determine:
1. Did the command succeed?
2. Does the output match expectations?
3. Any warnings or issues?

Respond with:
VALIDATION: PASSED or FAILED
REASON: Brief explanation""",
                )
                
                response = validation_result.get("response", "")
                validation_passed = "PASSED" in response.upper()
                self._state["validation_response"] = response
                self._current_step = 3
            
            # Write log
            log_path = self._write_log()
            
            self._status = ModuleStatus.COMPLETED
            
            return ModuleResult(
                status=ModuleStatus.COMPLETED,
                outputs={
                    "output": self._state.get("execution_result", ""),
                    "exit_code": 0 if validation_passed else 1,  # Inferred
                    "validation_passed": validation_passed,
                    "log_path": log_path,
                },
                artifacts=[log_path],
                summary=f"Check '{inputs['check_name']}' {'passed' if validation_passed else 'failed'}.",
            )
            
        except Exception as e:
            self._log_step(f"Error: {e}")
            return ModuleResult(status=ModuleStatus.FAILED, error=str(e))
    
    def _write_log(self) -> str:
        """Write check log."""
        logs_dir = self.project_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_name = self._state["check_name"].replace(" ", "_").lower()
        log_path = logs_dir / f"{log_name}.log"
        
        with open(log_path, "w") as f:
            f.write(f"# Check: {self._state['check_name']}\n")
            f.write(f"Command: {self._state['command']}\n")
            f.write(f"CWD: {self._state['cwd']}\n")
            f.write("-" * 50 + "\n")
            f.write("Output:\n")
            f.write(self._state.get("execution_result", ""))
            f.write("\n" + "-" * 50 + "\n")
            f.write("Validation:\n")
            f.write(self._state.get("validation_response", "Not performed"))
        
        return str(log_path)
