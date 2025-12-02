from __future__ import annotations

from typing import Any, Dict, List
from MiniLab.storage.state_store import StateStore, Citation

from . import Tool


class CitationIndexTool(Tool):
    """
    Interface with the local citation index for searching and managing references.
    """

    def __init__(self, state_store: StateStore):
        super().__init__(
            name="citation_index",
            description="Search and retrieve citations from the knowledge base"
        )
        self.state_store = state_store

    async def execute(
        self,
        action: str,
        query: str | None = None,
        project_id: str | None = None,
        citation_key: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute citation index operation.
        
        Args:
            action: Action to perform ("search", "get", "related")
            query: Search query
            project_id: Project context (optional)
            citation_key: Specific citation key to retrieve
            
        Returns:
            Dict with citation results
        """
        try:
            if action == "search":
                if not query:
                    return {"status": "error", "message": "query required for search"}
                
                citations = self.state_store.search_citations(query, project_id)
                return {
                    "status": "success",
                    "query": query,
                    "count": len(citations),
                    "citations": [self._format_citation(c) for c in citations[:20]]
                }
            
            elif action == "get":
                if not citation_key:
                    return {"status": "error", "message": "citation_key required"}
                
                if project_id:
                    project = self.state_store.load_project(project_id)
                    if project and citation_key in project.citations:
                        citation = project.citations[citation_key]
                        return {
                            "status": "success",
                            "citation": self._format_citation(citation)
                        }
                
                # Try global bibliography
                if citation_key in self.state_store.global_bibliography:
                    citation = self.state_store.global_bibliography[citation_key]
                    return {
                        "status": "success",
                        "citation": self._format_citation(citation)
                    }
                
                return {
                    "status": "error",
                    "message": f"Citation {citation_key} not found"
                }
            
            elif action == "related":
                # Find related citations based on concept links
                if not project_id:
                    return {"status": "error", "message": "project_id required for related"}
                
                project = self.state_store.load_project(project_id)
                if not project:
                    return {"status": "error", "message": f"Project {project_id} not found"}
                
                # Find concept links involving this citation
                related = []
                for link in project.concept_links:
                    if link.source == citation_key or link.target == citation_key:
                        other_key = link.target if link.source == citation_key else link.source
                        if other_key in project.citations:
                            related.append({
                                "citation": self._format_citation(project.citations[other_key]),
                                "relation": link.relation_type,
                                "description": link.description,
                            })
                
                return {
                    "status": "success",
                    "citation_key": citation_key,
                    "related": related
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _format_citation(self, citation: Citation) -> dict:
        """Format citation for display."""
        return {
            "key": citation.key,
            "title": citation.title,
            "authors": citation.authors,
            "year": citation.year,
            "journal": citation.journal,
            "doi": citation.doi,
            "url": citation.url,
            "tags": citation.tags,
            "notes": citation.notes,
        }


class GraphBuilderTool(Tool):
    """
    Build and query the knowledge graph of concepts and citations.
    """

    def __init__(self, state_store: StateStore):
        super().__init__(
            name="graph_builder",
            description="Build and query knowledge graphs of research concepts"
        )
        self.state_store = state_store

    async def execute(
        self,
        action: str,
        project_id: str,
        source: str | None = None,
        target: str | None = None,
        relation_type: str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute graph building operation.
        
        Args:
            action: Action to perform ("add_link", "get_graph", "find_paths")
            project_id: Project context
            source: Source node (citation key or concept)
            target: Target node
            relation_type: Type of relation
            
        Returns:
            Dict with graph data
        """
        try:
            project = self.state_store.load_project(project_id)
            if not project:
                return {"status": "error", "message": f"Project {project_id} not found"}

            if action == "add_link":
                if not all([source, target, relation_type]):
                    return {
                        "status": "error",
                        "message": "source, target, and relation_type required"
                    }
                
                from minilab.storage.state_store import ConceptLink
                link = ConceptLink(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    description=kwargs.get("description", ""),
                    strength=kwargs.get("strength", 1.0),
                )
                project.add_concept_link(link)
                self.state_store.save_project(project)
                
                return {
                    "status": "success",
                    "message": "Link added successfully"
                }
            
            elif action == "get_graph":
                # Return graph structure
                nodes = set()
                edges = []
                
                for link in project.concept_links:
                    nodes.add(link.source)
                    nodes.add(link.target)
                    edges.append({
                        "source": link.source,
                        "target": link.target,
                        "type": link.relation_type,
                        "description": link.description,
                        "strength": link.strength,
                    })
                
                return {
                    "status": "success",
                    "nodes": list(nodes),
                    "edges": edges,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                }
            
            elif action == "find_paths":
                # Simple path finding between two nodes
                if not all([source, target]):
                    return {"status": "error", "message": "source and target required"}
                
                # Build adjacency list
                graph = {}
                for link in project.concept_links:
                    if link.source not in graph:
                        graph[link.source] = []
                    graph[link.source].append((link.target, link.relation_type))
                
                # BFS to find shortest path
                from collections import deque
                queue = deque([(source, [source])])
                visited = {source}
                
                while queue:
                    node, path = queue.popleft()
                    
                    if node == target:
                        return {
                            "status": "success",
                            "path": path,
                            "length": len(path) - 1
                        }
                    
                    for neighbor, relation in graph.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                
                return {
                    "status": "success",
                    "path": None,
                    "message": "No path found"
                }
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
