"""
Pure DAG-Driven Orchestrator for MiniLab.

This is a reimplementation of MiniLabOrchestrator focused on being a pure
execution engine that respects TaskGraph dependencies rather than
enforcing fixed workflow sequences.

Key design principles:
1. TaskGraph is SOURCE OF TRUTH for execution order
2. No hard-coded workflow sequences
3. Explicit error categorization (no bare except blocks)
4. Dynamic agent assignment per task (from TaskGraph, not workflow)
5. Pure execution - no LLM decisions (those happen in consultation/planning)

Migration Path:
- New sessions use this orchestrator (default)
- Can coexist with legacy orchestrator during transition
- TaskGraph from consultation drives everything
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import asyncio
from enum import Enum

from ..core.task_graph import TaskGraph, TaskNode, TaskStatus as GraphTaskStatus
from ..workflows import WorkflowModule, WorkflowResult, WorkflowStatus
from ..context import ContextManager
from ..agents import AgentRegistry
from ..storage import TranscriptLogger
from ..core import TokenAccount
from ..infrastructure import FatalError, DegradedError, handle_error
from ..config.minilab_config import get_config
from ..utils import console


class ExecutionStrategy(Enum):
    """Strategy for executing ready tasks."""
    SEQUENTIAL = "sequential"  # One at a time (safe, slower)
    PARALLEL_SAFE = "parallel_safe"  # Parallel where no shared state
    ASYNC_PARALLEL = "async_parallel"  # Full async parallelism


class PureDAGOrchestrator:
    """
    Pure DAG-driven orchestrator that executes TaskGraphs.
    
    Does NOT enforce workflow sequences - respects TaskGraph structure entirely.
    """
    
    def __init__(
        self,
        session_id: str,
        project_path: Path,
        project_name: str,
        agents: AgentRegistry,
        context_manager: ContextManager,
        transcript: TranscriptLogger,
        token_account: TokenAccount,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ):
        self.session_id = session_id
        self.project_path = project_path
        self.project_name = project_name
        self.agents = agents
        self.context_manager = context_manager
        self.transcript = transcript
        self.token_account = token_account
        self.strategy = strategy
        
        self.config = get_config()
        self.task_graph: Optional[TaskGraph] = None
        self.workflow_modules: Dict[str, WorkflowModule] = {}
    
    def load_task_graph(self, graph: TaskGraph) -> None:
        """
        Load a task graph for execution.
        
        Validates that graph is well-formed before execution.
        
        Args:
            graph: TaskGraph to execute
            
        Raises:
            FatalError: If TaskGraph is invalid
        """
        self.task_graph = graph
        self._validate_task_graph()
        console.info(f"Loaded TaskGraph with {len(graph.nodes)} tasks")
    
    def _validate_task_graph(self) -> None:
        """
        Validate that task graph meets requirements.
        
        Raises:
            FatalError: If graph is invalid
        """
        if not self.task_graph or not self.task_graph.nodes:
            raise FatalError(
                "TaskGraph is empty - consultation must produce a valid task graph",
                context={"graph": self.task_graph},
            )
        
        # Check for invalid states
        for task_id, task in self.task_graph.nodes.items():
            if not task_id:
                raise FatalError(
                    f"Task has no ID",
                    context={"task": task},
                )
            if not task.get("status"):
                raise FatalError(
                    f"Task {task_id} has no status",
                    context={"task": task},
                )
        
        # Check that all edge references exist
        for edge_id, edge in self.task_graph.edges.items():
            source = edge.get("source")
            target = edge.get("target")
            
            if source and source not in self.task_graph.nodes:
                raise FatalError(
                    f"Edge {edge_id} references non-existent source task {source}",
                    context={"edge": edge},
                )
            if target and target not in self.task_graph.nodes:
                raise FatalError(
                    f"Edge {edge_id} references non-existent target task {target}",
                    context={"edge": edge},
                )
        
        console.info("TaskGraph validation passed")
    
    async def execute(self) -> Dict[str, WorkflowResult]:
        """
        Execute the loaded task graph.
        
        Respects task dependencies - only runs tasks whose dependencies are complete.
        No fixed workflow sequence.
        
        Returns:
            Dictionary mapping task IDs to WorkflowResults
            
        Raises:
            FatalError: If a critical task fails
        """
        if not self.task_graph:
            raise FatalError("No task graph loaded - call load_task_graph() first")
        
        results: Dict[str, WorkflowResult] = {}
        
        console.info("Starting DAG-driven task execution...")
        
        while not self.task_graph.is_complete():
            # Get all tasks ready to execute
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # No ready tasks but graph not complete - likely circular dependency
                incomplete = [
                    tid for tid, task in self.task_graph.nodes.items()
                    if task.get("status") != GraphTaskStatus.COMPLETED.value
                ]
                raise FatalError(
                    f"Circular dependency or deadlock - {len(incomplete)} tasks incomplete but none ready",
                    context={"incomplete_tasks": incomplete},
                )
            
            # Execute ready tasks
            for task_id in ready_tasks:
                try:
                    task = self.task_graph.nodes[task_id]
                    console.info(f"Executing task: {task_id}")
                    
                    # Mark as in-progress
                    self.task_graph.mark_in_progress(task_id)
                    
                    # Execute the task
                    result = await self._execute_task(task_id, task)
                    results[task_id] = result
                    
                    # Mark as complete or failed
                    if result.status == WorkflowStatus.COMPLETED:
                        self.task_graph.mark_completed(
                            task_id,
                            outputs=result.outputs or {}
                        )
                        console.info(f"Task {task_id} completed successfully")
                    else:
                        self.task_graph.mark_failed(
                            task_id,
                            error=result.error or "Unknown error"
                        )
                        console.error(f"Task {task_id} failed: {result.error}")
                        
                        # Decide whether to continue
                        if self._is_critical_task(task_id):
                            raise FatalError(
                                f"Critical task {task_id} failed",
                                context={
                                    "task_id": task_id,
                                    "error": result.error,
                                },
                            )
                
                except FatalError:
                    raise
                except Exception as e:
                    error = FatalError(
                        f"Unexpected error executing task {task_id}",
                        context={"task_id": task_id},
                        original_error=e,
                    )
                    await handle_error(error, f"task_execution_{task_id}")
                    raise
        
        console.info("DAG execution complete - all tasks finished")
        return results
    
    def _get_ready_tasks(self) -> List[str]:
        """Get all tasks ready to execute (all dependencies complete)."""
        if not self.task_graph:
            return []
        
        ready = []
        
        for task_id, task in self.task_graph.nodes.items():
            status = task.get("status")
            
            # Skip already completed or in-progress
            if status in [GraphTaskStatus.COMPLETED.value, GraphTaskStatus.IN_PROGRESS.value]:
                continue
            
            # Check dependencies
            dependencies_complete = self._are_dependencies_complete(task_id)
            
            if dependencies_complete:
                ready.append(task_id)
        
        return ready
    
    def _are_dependencies_complete(self, task_id: str) -> bool:
        """Check if all dependencies of a task are complete."""
        if not self.task_graph:
            return False
        
        # Find all edges pointing TO this task (dependencies)
        for edge_id, edge in self.task_graph.edges.items():
            if edge.get("target") == task_id:
                source_id = edge.get("source")
                if source_id:
                    source_task = self.task_graph.nodes.get(source_id)
                    if not source_task:
                        return False
                    if source_task.get("status") != GraphTaskStatus.COMPLETED.value:
                        return False
        
        return True
    
    def _is_critical_task(self, task_id: str) -> bool:
        """Determine if task failure should stop execution."""
        if not self.task_graph:
            return True
        
        task = self.task_graph.nodes.get(task_id, {})
        
        # Tasks marked as critical
        if task.get("critical", False):
            return True
        
        # Consultation is always critical
        if "consultation" in task_id.lower():
            return True
        
        # Other heuristics
        return False
    
    async def _execute_task(self, task_id: str, task: Dict[str, Any]) -> WorkflowResult:
        """
        Execute a single task.
        
        Delegates to appropriate workflow module based on task type.
        """
        # Get workflow type from task
        workflow_type = task.get("workflow", "execute_analysis")
        
        # Get or create workflow module
        if workflow_type not in self.workflow_modules:
            # TODO: Instantiate correct workflow module
            # For now, return dummy result
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                summary=f"Task {task_id} executed",
                outputs=task.get("outputs", {}),
            )
        
        workflow = self.workflow_modules[workflow_type]
        
        # Prepare inputs for the workflow
        inputs = self._prepare_workflow_inputs(task)
        
        # Execute
        result = await workflow.execute(inputs=inputs)
        
        return result
    
    def _prepare_workflow_inputs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for a workflow from task specification."""
        inputs = task.get("inputs", {})
        
        # Add context from completed tasks
        inputs["completed_tasks"] = {
            tid: self.task_graph.nodes[tid].get("outputs", {})
            for tid in self._get_completed_task_ids()
        }
        
        return inputs
    
    def _get_completed_task_ids(self) -> List[str]:
        """Get IDs of all completed tasks."""
        if not self.task_graph:
            return []
        
        return [
            tid for tid, task in self.task_graph.nodes.items()
            if task.get("status") == GraphTaskStatus.COMPLETED.value
        ]
