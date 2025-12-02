from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx

from . import Tool


class ZoteroTool(Tool):
    """
    Interface with Zotero library using the Zotero Web API.
    Requires Zotero API key and user/group ID.
    
    Documentation: https://www.zotero.org/support/dev/web_api/v3/start
    """

    def __init__(
        self,
        api_key: str | None = None,
        library_type: str = "user",  # "user" or "group"
        library_id: str | None = None,
    ):
        super().__init__(
            name="zotero",
            description="Access and manage Zotero library for literature management"
        )
        self.api_key = api_key or os.environ.get("ZOTERO_API_KEY")
        self.library_type = library_type
        self.library_id = library_id or os.environ.get("ZOTERO_USER_ID")
        
        if not self.api_key or not self.library_id:
            raise RuntimeError(
                "Zotero requires ZOTERO_API_KEY and ZOTERO_USER_ID environment variables"
            )
        
        self.client = httpx.AsyncClient(
            base_url="https://api.zotero.org",
            headers={
                "Zotero-API-Key": self.api_key,
                "Zotero-API-Version": "3",
            },
            timeout=30.0,
        )

    def _get_base_path(self) -> str:
        """Get base API path for the library."""
        return f"/{self.library_type}s/{self.library_id}"

    async def execute(
        self,
        action: str,
        collection_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a Zotero action.
        
        Args:
            action: Action to perform ("list_collections", "list_items", "search", "get_item")
            collection_id: Collection ID for collection-specific operations
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dict with results
        """
        try:
            if action == "list_collections":
                return await self._list_collections()
            
            elif action == "list_items":
                return await self._list_items(collection_id, limit)
            
            elif action == "search":
                return await self._search_items(query, limit)
            
            elif action == "get_item":
                item_key = kwargs.get("item_key")
                if not item_key:
                    return {"status": "error", "message": "item_key required for get_item"}
                return await self._get_item(item_key)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}. Supported: list_collections, list_items, search, get_item"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    async def _list_collections(self) -> Dict[str, Any]:
        """List all collections in the library."""
        path = f"{self._get_base_path()}/collections"
        response = await self.client.get(path)
        response.raise_for_status()
        collections = response.json()
        
        return {
            "status": "success",
            "collections": [
                {
                    "key": c["key"],
                    "name": c["data"]["name"],
                    "parent": c["data"].get("parentCollection", None),
                }
                for c in collections
            ]
        }

    async def _list_items(
        self,
        collection_id: Optional[str] = None,
        limit: int = 25
    ) -> Dict[str, Any]:
        """List items in library or specific collection."""
        if collection_id:
            path = f"{self._get_base_path()}/collections/{collection_id}/items"
        else:
            path = f"{self._get_base_path()}/items"
        
        response = await self.client.get(path, params={"limit": limit})
        response.raise_for_status()
        items = response.json()
        
        return {
            "status": "success",
            "items": [self._format_item(item) for item in items]
        }

    async def _search_items(self, query: str, limit: int = 25) -> Dict[str, Any]:
        """Search for items in the library."""
        path = f"{self._get_base_path()}/items"
        params = {
            "q": query,
            "limit": limit,
            "qmode": "everything",  # search all fields
        }
        
        response = await self.client.get(path, params=params)
        response.raise_for_status()
        items = response.json()
        
        return {
            "status": "success",
            "query": query,
            "items": [self._format_item(item) for item in items]
        }

    async def _get_item(self, item_key: str) -> Dict[str, Any]:
        """Get a specific item by key."""
        path = f"{self._get_base_path()}/items/{item_key}"
        response = await self.client.get(path)
        response.raise_for_status()
        item = response.json()
        
        return {
            "status": "success",
            "item": self._format_item(item)
        }

    def _format_item(self, item: dict) -> dict:
        """Format a Zotero item for easier consumption."""
        data = item.get("data", {})
        return {
            "key": item.get("key"),
            "type": data.get("itemType"),
            "title": data.get("title", ""),
            "creators": data.get("creators", []),
            "date": data.get("date", ""),
            "doi": data.get("DOI", ""),
            "url": data.get("url", ""),
            "abstract": data.get("abstractNote", ""),
            "tags": [t.get("tag") for t in data.get("tags", [])],
            "collections": data.get("collections", []),
        }
