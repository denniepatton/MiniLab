"""Token context metadata with universal taxonomy.

Attaches structured taxonomy metadata (module, op_kind, tool_family) to
TokenAccount debits using contextvars. This replaces the old workflow/trigger
strings with a universal, learnable taxonomy.

The taxonomy enables:
- Meaningful aggregation across projects (modules are stable)
- Fine-grained cost tracking (op_kind, tool_family)
- Actionable budget guidance (learned estimates per category)

Legacy workflow/trigger strings are still supported for backward compatibility
but are converted to taxonomy on use.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .taxonomy import Module, OpKind, ToolFamily, TaxonomyContext


# New taxonomy-based context vars
_current_module: ContextVar[Optional["Module"]] = ContextVar("minilab_current_module", default=None)
_current_op_kind: ContextVar[Optional["OpKind"]] = ContextVar("minilab_current_op_kind", default=None)
_current_tool_family: ContextVar[Optional["ToolFamily"]] = ContextVar("minilab_current_tool_family", default=None)
_current_agent_id: ContextVar[Optional[str]] = ContextVar("minilab_current_agent_id", default=None)

# Legacy context vars (kept for backward compatibility)
_current_workflow: ContextVar[Optional[str]] = ContextVar("minilab_current_workflow", default=None)
_current_trigger: ContextVar[Optional[str]] = ContextVar("minilab_current_trigger", default=None)


# =============================================================================
# Taxonomy Getters
# =============================================================================

def get_module() -> Optional["Module"]:
    """Get current module from context."""
    return _current_module.get()


def get_op_kind() -> Optional["OpKind"]:
    """Get current operation kind from context."""
    return _current_op_kind.get()


def get_tool_family() -> Optional["ToolFamily"]:
    """Get current tool family from context."""
    return _current_tool_family.get()


def get_agent_id() -> Optional[str]:
    """Get current agent ID from context."""
    return _current_agent_id.get()


def get_taxonomy_context() -> "TaxonomyContext":
    """
    Get complete taxonomy context from current contextvars.
    
    If new taxonomy vars are set, uses those directly.
    If only legacy vars are set, converts them to taxonomy.
    """
    from .taxonomy import TaxonomyContext, Module, OpKind, ToolFamily
    
    module = _current_module.get()
    op_kind = _current_op_kind.get()
    tool_family = _current_tool_family.get()
    agent_id = _current_agent_id.get()
    
    # Get legacy values
    workflow = _current_workflow.get()
    trigger = _current_trigger.get()
    
    # If we have explicit taxonomy values, use them
    if module is not None or op_kind is not None or tool_family is not None:
        return TaxonomyContext(
            module=module,
            op_kind=op_kind,
            tool_family=tool_family,
            agent_id=agent_id,
            legacy_workflow=workflow,
            legacy_trigger=trigger,
        )
    
    # Otherwise, try to parse from legacy values
    if workflow or trigger:
        ctx = TaxonomyContext.from_legacy(workflow, trigger)
        ctx.agent_id = agent_id
        return ctx
    
    # No context available
    return TaxonomyContext(agent_id=agent_id)


# =============================================================================
# Legacy Getters (backward compatibility)
# =============================================================================

def get_workflow() -> Optional[str]:
    """Get current workflow string (legacy, prefer get_module())."""
    return _current_workflow.get()


def get_trigger() -> Optional[str]:
    """Get current trigger string (legacy, prefer get_op_kind())."""
    return _current_trigger.get()


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def taxonomy_context(
    *,
    module: Optional["Module"] = None,
    op_kind: Optional["OpKind"] = None,
    tool_family: Optional["ToolFamily"] = None,
    agent_id: Optional[str] = None,
) -> Iterator[None]:
    """
    Set taxonomy context for token attribution.
    
    This is the preferred way to set context. Use this instead of
    token_context() for new code.
    
    Example:
        from MiniLab.core.taxonomy import Module, OpKind, ToolFamily
        
        with taxonomy_context(
            module=Module.LITERATURE_REVIEW,
            op_kind=OpKind.RETRIEVE,
            tool_family=ToolFamily.SEARCH,
            agent_id="gould",
        ):
            result = await agent.run(task)
    """
    tokens = []
    if module is not None:
        tokens.append((_current_module, _current_module.set(module)))
    if op_kind is not None:
        tokens.append((_current_op_kind, _current_op_kind.set(op_kind)))
    if tool_family is not None:
        tokens.append((_current_tool_family, _current_tool_family.set(tool_family)))
    if agent_id is not None:
        tokens.append((_current_agent_id, _current_agent_id.set(agent_id)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass


@contextmanager
def token_context(
    *,
    workflow: Optional[str] = None,
    trigger: Optional[str] = None,
) -> Iterator[None]:
    """
    Set legacy token context (workflow/trigger strings).
    
    DEPRECATED: Use taxonomy_context() for new code.
    
    This function remains for backward compatibility. It sets the legacy
    context vars, and get_taxonomy_context() will convert them to taxonomy.
    """
    tokens = []
    if workflow is not None:
        tokens.append((_current_workflow, _current_workflow.set(workflow)))
    if trigger is not None:
        tokens.append((_current_trigger, _current_trigger.set(trigger)))
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass


@contextmanager
def tool_context(
    *,
    tool_name: str,
    action: str,
    agent_id: Optional[str] = None,
) -> Iterator[None]:
    """
    Set context for a tool operation.
    
    Automatically infers op_kind and tool_family from tool name and action.
    
    Example:
        with tool_context(tool_name="pubmed", action="search", agent_id="gould"):
            result = await pubmed_tool.execute("search", params)
    """
    from .taxonomy import OpKind, ToolFamily
    
    op_kind = OpKind.from_tool_action(tool_name, action)
    tool_family = ToolFamily.from_tool_name(tool_name)
    
    tokens = [
        (_current_op_kind, _current_op_kind.set(op_kind)),
        (_current_tool_family, _current_tool_family.set(tool_family)),
        (_current_trigger, _current_trigger.set(f"after_tool:{tool_name}.{action}")),
    ]
    if agent_id is not None:
        tokens.append((_current_agent_id, _current_agent_id.set(agent_id)))
    
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass


@contextmanager
def module_context(module: "Module", agent_id: Optional[str] = None) -> Iterator[None]:
    """
    Set module context for a workflow execution.
    
    Use this when entering a workflow module to properly attribute all
    subsequent token usage.
    
    Example:
        with module_context(Module.LITERATURE_REVIEW, agent_id="gould"):
            await literature_review_workflow.execute(inputs)
    """
    tokens = [
        (_current_module, _current_module.set(module)),
        (_current_workflow, _current_workflow.set(module.value)),
    ]
    if agent_id is not None:
        tokens.append((_current_agent_id, _current_agent_id.set(agent_id)))
    
    try:
        yield
    finally:
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass
