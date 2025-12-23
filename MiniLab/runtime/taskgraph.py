"""
TaskGraph: DAG of TaskNodes for work planning.

Provides:
- TaskNode: Individual work unit with owner, dependencies, and schema
- TaskGraph: Collection of nodes with topological ordering
- TaskStatus: Execution state tracking
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TaskStatus(str, Enum):
    """Status of a task node."""
    PENDING = "pending"
    READY = "ready"  # All dependencies satisfied
    RUNNING = "running"
    BLOCKED = "blocked"  # Waiting on failed dependency
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """
    A single task in the execution graph.
    
    Attributes:
        id: Unique identifier for this task
        description: Human-readable description of what the task does
        owner_agent: Agent persona responsible for this task
        required_tools: Tools needed to complete this task
        inputs: Input data/context for the task
        outputs_schema: JSON schema for expected outputs
        status: Current execution status
        retries: Number of retry attempts remaining
        max_retries: Maximum retry attempts
        depends_on: IDs of tasks this depends on
        result: Task output after completion
        error: Error message if failed
        started_at: When execution started
        completed_at: When execution completed
    """

    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="What this task does")
    owner_agent: str = Field(..., description="Agent responsible for this task")
    required_tools: list[str] = Field(default_factory=list, description="Tools needed")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input data")
    outputs_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object"},
        description="JSON schema for outputs"
    )

    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    retries: int = Field(default=3, description="Retries remaining")
    max_retries: int = Field(default=3, description="Maximum retries")
    depends_on: list[str] = Field(default_factory=list, description="Dependency task IDs")

    result: Optional[dict[str, Any]] = Field(default=None, description="Task output")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")

    model_config = {"extra": "forbid"}

    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_done(self, result: dict[str, Any]) -> None:
        """Mark task as completed successfully."""
        self.status = TaskStatus.DONE
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retries > 0

    def decrement_retries(self) -> None:
        """Decrement retry counter."""
        if self.retries > 0:
            self.retries -= 1


class TaskGraph(BaseModel):
    """
    A directed acyclic graph of tasks.
    
    Manages task dependencies and execution ordering.
    """

    nodes: dict[str, TaskNode] = Field(default_factory=dict, description="Task nodes by ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Graph metadata")

    model_config = {"extra": "forbid"}

    def add_node(self, node: TaskNode) -> None:
        """Add a task node to the graph."""
        if node.id in self.nodes:
            raise ValueError(f"Task with ID '{node.id}' already exists")
        self.nodes[node.id] = node

    def get_node(self, task_id: str) -> Optional[TaskNode]:
        """Get a task node by ID."""
        return self.nodes.get(task_id)

    def topological_order(self) -> list[str]:
        """
        Return task IDs in topological order (respecting dependencies).
        
        Raises:
            ValueError: If graph contains cycles
        """
        # Kahn's algorithm
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}

        # Calculate in-degrees
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                if dep_id in in_degree:
                    in_degree[node.id] = in_degree.get(node.id, 0)
                    # This dependency adds to our in-degree
                    pass

        # Actually calculate: for each node, how many nodes depend on it?
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                if dep_id not in self.nodes:
                    raise ValueError(f"Task '{node.id}' depends on unknown task '{dep_id}'")

        # Reset and recalculate properly
        in_degree = {node_id: len(self.nodes[node_id].depends_on) for node_id in self.nodes}

        # Start with nodes that have no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for nodes that depend on current
            for node_id, node in self.nodes.items():
                if current in node.depends_on:
                    in_degree[node_id] -= 1
                    if in_degree[node_id] == 0:
                        queue.append(node_id)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")

        return result

    def ready_nodes(self) -> list[TaskNode]:
        """
        Return nodes that are ready to execute.
        
        A node is ready if:
        - Status is PENDING
        - All dependencies are DONE
        """
        ready = []

        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are done
            deps_satisfied = all(
                self.nodes[dep_id].status == TaskStatus.DONE
                for dep_id in node.depends_on
                if dep_id in self.nodes
            )

            if deps_satisfied:
                node.status = TaskStatus.READY
                ready.append(node)

        return ready

    def blocked_nodes(self) -> list[TaskNode]:
        """Return nodes that are blocked due to failed dependencies."""
        blocked = []

        for node in self.nodes.values():
            if node.status in (TaskStatus.PENDING, TaskStatus.READY):
                # Check if any dependency failed
                for dep_id in node.depends_on:
                    if dep_id in self.nodes:
                        dep = self.nodes[dep_id]
                        if dep.status == TaskStatus.FAILED:
                            node.status = TaskStatus.BLOCKED
                            blocked.append(node)
                            break

        return blocked

    def is_complete(self) -> bool:
        """Check if all tasks are in terminal state."""
        terminal_states = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.BLOCKED}
        return all(node.status in terminal_states for node in self.nodes.values())

    def success_count(self) -> int:
        """Count successfully completed tasks."""
        return sum(1 for node in self.nodes.values() if node.status == TaskStatus.DONE)

    def failure_count(self) -> int:
        """Count failed tasks."""
        return sum(1 for node in self.nodes.values() if node.status == TaskStatus.FAILED)

    def summary(self) -> dict[str, Any]:
        """Get execution summary."""
        status_counts = {}
        for node in self.nodes.values():
            status_counts[node.status.value] = status_counts.get(node.status.value, 0) + 1

        return {
            "total_tasks": len(self.nodes),
            "status_counts": status_counts,
            "is_complete": self.is_complete(),
            "success_rate": self.success_count() / len(self.nodes) if self.nodes else 0,
        }
