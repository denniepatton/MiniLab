"""
Context Manager - Orchestrates RAG retrieval and context building.

Implements the structured context philosophy:
1. Static header (persona, role, objective, tools)
2. Rolling task state (~1000 tokens, refreshed via summarization)
3. Targeted external context (RAG retrieval with re-ranking)
4. Canonical state objects (tasks, plans, decisions)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .embeddings import EmbeddingManager
from .vector_store import VectorStore, Document, SearchResult
from .state_objects import (
    ProjectState,
    TaskState,
    ConversationSummary,
    WorkingPlan,
    ExecutionPlan,
    DataManifest,
)


@dataclass
class AgentHeader:
    """
    Static header for agent context.
    
    NOT updated after initialization within a specific project.
    """
    agent_id: str
    persona: str
    role: str
    objective: str
    tools_documentation: str
    
    def to_context(self) -> str:
        """Generate header context string."""
        return f"""# Agent: {self.agent_id.upper()}

## Persona
{self.persona}

## Role
{self.role}

## Project Objective
{self.objective}

{self.tools_documentation}
"""


@dataclass
class RetrievedChunk:
    """A chunk retrieved from RAG with metadata."""
    content: str
    source: str  # File path or document ID
    chunk_type: str  # 'code', 'document', 'conversation', 'literature'
    relevance_score: float
    recency_score: float
    compressed: bool = False


@dataclass
class ProjectContext:
    """
    Complete context for an agent on a project.
    
    This is passed to the LLM for each request.
    """
    header: AgentHeader
    task_state: TaskState
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    project_state_summary: str = ""
    
    # Token budget tracking
    header_tokens: int = 0
    task_state_tokens: int = 0
    retrieved_tokens: int = 0
    total_tokens: int = 0
    
    TARGET_TOKENS = 8000  # Target total context size
    TASK_STATE_MAX = 1000
    RETRIEVED_MAX = 4000
    
    def to_prompt(self) -> str:
        """Generate the full context prompt."""
        sections = [self.header.to_context()]
        
        if self.task_state:
            sections.append(self.task_state.to_context())
        
        if self.project_state_summary:
            sections.append(f"## Project State\n{self.project_state_summary}")
        
        if self.retrieved_chunks:
            sections.append("## Retrieved Context")
            for chunk in self.retrieved_chunks:
                sections.append(f"\n### {chunk.source} ({chunk.chunk_type})")
                sections.append(chunk.content)
        
        return "\n\n".join(sections)


class ContextManager:
    """
    Manages context for all agents across projects.
    
    Features:
    - Project-specific context persistence
    - RAG retrieval with semantic + recency scoring
    - Automatic context compression
    - Async context rebuilding
    """
    
    def __init__(
        self,
        workspace_root: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieval_k: int = 10,
        recency_weight: float = 0.2,
    ):
        """
        Initialize context manager.
        
        Args:
            workspace_root: Root of the workspace
            embedding_model: sentence-transformers model name
            retrieval_k: Number of chunks to retrieve
            recency_weight: Weight for recency in scoring
        """
        self.workspace_root = workspace_root
        self.retrieval_k = retrieval_k
        self.recency_weight = recency_weight
        
        # Initialize embedding manager
        cache_dir = workspace_root / "Sandbox" / ".context_cache"
        self.embeddings = EmbeddingManager(
            model_name=embedding_model,
            cache_dir=cache_dir,
        )
        
        # Project-specific vector stores
        self._stores: dict[str, VectorStore] = {}
        
        # Project states
        self._project_states: dict[str, ProjectState] = {}
        
        # Agent headers (cached)
        self._agent_headers: dict[str, AgentHeader] = {}
        
        # Background task for async context updates
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._background_task: Optional[asyncio.Task] = None
    
    def _get_store(self, project_name: str) -> VectorStore:
        """Get or create vector store for a project."""
        if project_name not in self._stores:
            persist_dir = self.workspace_root / "Sandbox" / project_name / ".context"
            self._stores[project_name] = VectorStore(
                embedding_dim=self.embeddings.embedding_dim,
                persist_dir=persist_dir,
                recency_weight=self.recency_weight,
            )
        return self._stores[project_name]
    
    def set_agent_header(
        self,
        agent_id: str,
        persona: str,
        role: str,
        objective: str,
        tools_documentation: str,
    ) -> None:
        """
        Set the static header for an agent.
        
        This should be called once at project initialization.
        """
        self._agent_headers[agent_id] = AgentHeader(
            agent_id=agent_id,
            persona=persona,
            role=role,
            objective=objective,
            tools_documentation=tools_documentation,
        )
    
    def get_agent_header(self, agent_id: str) -> Optional[AgentHeader]:
        """Get the cached header for an agent."""
        return self._agent_headers.get(agent_id)
    
    def load_project_state(self, project_name: str) -> Optional[ProjectState]:
        """Load project state from disk."""
        state_path = self.workspace_root / "Sandbox" / project_name / "project_state.json"
        
        if state_path.exists():
            state = ProjectState.load(str(state_path))
            self._project_states[project_name] = state
            return state
        
        return None
    
    def save_project_state(self, project_name: str) -> None:
        """Save project state to disk."""
        if project_name in self._project_states:
            state = self._project_states[project_name]
            state.save()
    
    def get_project_state(self, project_name: str) -> Optional[ProjectState]:
        """Get project state, loading if necessary."""
        if project_name not in self._project_states:
            self.load_project_state(project_name)
        return self._project_states.get(project_name)
    
    def set_project_state(self, project_name: str, state: ProjectState) -> None:
        """Set project state."""
        self._project_states[project_name] = state
    
    def index_document(
        self,
        project_name: str,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Index a document for RAG retrieval.
        
        Args:
            project_name: Project to index for
            doc_id: Unique document ID
            content: Document content
            metadata: Document metadata (type, source, etc.)
        """
        store = self._get_store(project_name)
        
        # Generate embedding
        embedding = self.embeddings.embed(content)
        
        # Create and add document
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
        )
        store.add(doc)
    
    def index_file(
        self,
        project_name: str,
        file_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> int:
        """
        Index a file for RAG retrieval.
        
        Args:
            project_name: Project to index for
            file_path: Path to file
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks indexed
        """
        if not file_path.exists():
            return 0
        
        content = file_path.read_text()
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{file_path.name}_{i}"
            self.index_document(
                project_name=project_name,
                doc_id=doc_id,
                content=chunk,
                metadata={
                    "type": "file",
                    "source": str(file_path),
                    "chunk_index": i,
                    "file_type": file_path.suffix,
                },
            )
        
        return len(chunks)
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def retrieve(
        self,
        project_name: str,
        query: str,
        k: Optional[int] = None,
        filter_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            project_name: Project to search
            query: Search query
            k: Number of results (default: self.retrieval_k)
            filter_type: Filter by document type
            
        Returns:
            List of retrieved chunks
        """
        store = self._get_store(project_name)
        k = k or self.retrieval_k
        
        # Generate query embedding
        query_embedding = self.embeddings.embed(query)
        
        # Build metadata filter
        metadata_filter = {}
        if filter_type:
            metadata_filter["type"] = filter_type
        
        # Search
        results = store.search(
            query_embedding=query_embedding,
            k=k,
            filter_metadata=metadata_filter if metadata_filter else None,
        )
        
        # Convert to RetrievedChunk
        chunks = []
        for result in results:
            chunks.append(RetrievedChunk(
                content=result.document.content,
                source=result.document.metadata.get("source", result.document.id),
                chunk_type=result.document.metadata.get("type", "unknown"),
                relevance_score=result.semantic_score,
                recency_score=result.recency_score,
            ))
        
        return chunks
    
    def compress_chunk(self, chunk: RetrievedChunk, max_tokens: int = 500) -> RetrievedChunk:
        """
        Compress a chunk to fit within token budget.
        
        Uses simple truncation with bullet point extraction.
        """
        content = chunk.content
        
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= max_tokens:
            return chunk
        
        # Extract key sentences/bullets
        lines = content.split("\n")
        important_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep headers, bullets, and short sentences
            if (
                line.startswith("#") or
                line.startswith("-") or
                line.startswith("*") or
                line.startswith("•") or
                len(line) < 100
            ):
                important_lines.append(line)
        
        compressed = "\n".join(important_lines)
        
        # If still too long, truncate
        max_chars = max_tokens * 4
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars] + "..."
        
        return RetrievedChunk(
            content=compressed,
            source=chunk.source,
            chunk_type=chunk.chunk_type,
            relevance_score=chunk.relevance_score,
            recency_score=chunk.recency_score,
            compressed=True,
        )
    
    def build_context(
        self,
        agent_id: str,
        project_name: str,
        query: str,
        include_state: bool = True,
    ) -> ProjectContext:
        """
        Build complete context for an agent request.
        
        Args:
            agent_id: Agent making the request
            project_name: Current project
            query: The query/task for context retrieval
            include_state: Whether to include project state
            
        Returns:
            Complete ProjectContext
        """
        # Get agent header
        header = self._agent_headers.get(agent_id)
        if not header:
            # Create minimal header
            header = AgentHeader(
                agent_id=agent_id,
                persona="Research assistant",
                role="General agent",
                objective="Assist with project tasks",
                tools_documentation="",
            )
        
        # Get project state
        project_state = self.get_project_state(project_name)
        task_state = project_state.task_state if project_state else TaskState()
        
        # Retrieve relevant chunks
        chunks = self.retrieve(project_name, query)
        
        # Compress chunks to fit budget
        compressed_chunks = []
        total_chunk_tokens = 0
        max_chunk_tokens = ProjectContext.RETRIEVED_MAX
        
        for chunk in chunks:
            remaining_tokens = max_chunk_tokens - total_chunk_tokens
            if remaining_tokens <= 100:
                break
            
            compressed = self.compress_chunk(chunk, max_tokens=remaining_tokens // len(chunks))
            compressed_chunks.append(compressed)
            total_chunk_tokens += len(compressed.content) // 4
        
        # Build project state summary
        state_summary = ""
        if include_state and project_state:
            summaries = []
            
            if project_state.data_manifest:
                summaries.append(project_state.data_manifest.to_context())
            
            if project_state.working_plan:
                summaries.append(project_state.working_plan.to_context())
            
            if project_state.execution_plan:
                summaries.append(project_state.execution_plan.to_context())
            
            state_summary = "\n\n".join(summaries)
        
        return ProjectContext(
            header=header,
            task_state=task_state,
            retrieved_chunks=compressed_chunks,
            project_state_summary=state_summary,
        )
    
    async def update_context_async(
        self,
        project_name: str,
        updates: dict[str, Any],
    ) -> None:
        """
        Queue a context update for async processing.
        
        This is the "sidecar" pattern - context updates happen
        in the background without blocking agent execution.
        """
        await self._update_queue.put((project_name, updates))
    
    async def _background_updater(self) -> None:
        """Background task for processing context updates."""
        while True:
            try:
                project_name, updates = await asyncio.wait_for(
                    self._update_queue.get(),
                    timeout=1.0,
                )
                
                # Process updates
                if "index_file" in updates:
                    self.index_file(project_name, Path(updates["index_file"]))
                
                if "index_document" in updates:
                    doc = updates["index_document"]
                    self.index_document(
                        project_name,
                        doc["id"],
                        doc["content"],
                        doc.get("metadata", {}),
                    )
                
                if "save_state" in updates:
                    self.save_project_state(project_name)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # Log error but continue
                print(f"Context update error: {e}")
    
    def start_background_updater(self) -> None:
        """Start the background context updater."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._background_updater())
    
    def stop_background_updater(self) -> None:
        """Stop the background context updater."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
    
    def summarize_conversation(
        self,
        messages: list[dict],
        existing_summary: Optional[ConversationSummary] = None,
    ) -> ConversationSummary:
        """
        Summarize conversation history into structured summary.
        
        This is called periodically to compress conversation history.
        
        Note: In production, this would call an LLM for summarization.
        For now, we do simple extraction.
        """
        # Extract key elements from messages
        action_items = []
        agreements = []
        questions = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            # Look for action items
            if "TODO" in content or "will do" in content.lower() or "should" in content.lower():
                # Extract the sentence
                for sentence in content.split("."):
                    if "TODO" in sentence or "will" in sentence.lower() or "should" in sentence.lower():
                        action_items.append(sentence.strip()[:100])
            
            # Look for agreements
            if "agreed" in content.lower() or "confirmed" in content.lower():
                for sentence in content.split("."):
                    if "agreed" in sentence.lower() or "confirmed" in sentence.lower():
                        agreements.append(sentence.strip()[:100])
            
            # Look for questions
            if "?" in content:
                for sentence in content.split("?"):
                    if len(sentence.strip()) > 10:
                        questions.append(sentence.strip()[:100] + "?")
        
        # Build summary
        summary_parts = []
        if existing_summary:
            summary_parts.append(existing_summary.summary)
        
        if messages:
            last_few = messages[-5:]
            summary_parts.append(f"Recent exchanges covered {len(last_few)} messages.")
        
        return ConversationSummary(
            summary=" ".join(summary_parts),
            action_items=action_items[-5:],
            agreements=agreements[-5:],
            open_questions=questions[-3:],
        )
