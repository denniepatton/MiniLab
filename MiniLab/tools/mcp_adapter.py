"""
MCP Adapters: Model Context Protocol integration.

Provides:
- MCPServer base class for exposing MiniLab tools via MCP
- MCPClient for consuming external MCP servers
- Tool bridging between MiniLab and MCP formats
"""

from __future__ import annotations

__all__ = [
    "MCPMessageType",
    "MCPMethod",
    "MCPToolSchema",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "MCPToolAdapter",
    "MCPServer",
    "MCPClient",
    "MCPToolBridge",
]

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from MiniLab.tools.base import Tool, ToolInput, ToolOutput


class MCPMessageType(str, Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(str, Enum):
    """MCP standard methods."""
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


class MCPToolSchema(BaseModel):
    """MCP tool definition schema."""

    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    inputSchema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for inputs"
    )

    model_config = {"extra": "allow"}


class MCPRequest(BaseModel):
    """MCP JSON-RPC request."""

    jsonrpc: str = Field(default="2.0")
    id: Optional[str] = None
    method: str
    params: Optional[dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class MCPResponse(BaseModel):
    """MCP JSON-RPC response."""

    jsonrpc: str = Field(default="2.0")
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None

    model_config = {"extra": "forbid"}


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


# Standard MCP error codes
MCP_PARSE_ERROR = -32700
MCP_INVALID_REQUEST = -32600
MCP_METHOD_NOT_FOUND = -32601
MCP_INVALID_PARAMS = -32602
MCP_INTERNAL_ERROR = -32603


class MCPToolAdapter:
    """
    Adapts MiniLab Tool to MCP tool format.
    
    Converts between MiniLab's Tool interface and MCP's tool protocol.
    """

    def __init__(self, tool: Tool):
        """
        Initialize adapter for a MiniLab tool.
        
        Args:
            tool: The MiniLab Tool instance to adapt
        """
        self.tool = tool

    def to_mcp_schema(self) -> list[MCPToolSchema]:
        """
        Convert tool actions to MCP tool schemas.
        
        Returns:
            List of MCP tool schemas (one per action)
        """
        schemas = []
        actions = self.tool.get_actions()

        for action_name, description in actions.items():
            # Get input schema for this action
            input_class = self.tool.get_input_schema(action_name)

            # Convert Pydantic model to JSON schema
            if hasattr(input_class, "model_json_schema"):
                input_schema = input_class.model_json_schema()
            else:
                input_schema = {"type": "object", "properties": {}}

            schemas.append(MCPToolSchema(
                name=f"{self.tool.name}.{action_name}",
                description=description,
                inputSchema=input_schema,
            ))

        return schemas

    async def execute_mcp(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute via MCP interface.
        
        Args:
            tool_name: Full tool name (tool.action)
            arguments: MCP arguments
            
        Returns:
            MCP-formatted result
        """
        # Parse tool.action format
        parts = tool_name.split(".", 1)
        if len(parts) != 2:
            raise MCPError(
                MCP_INVALID_PARAMS,
                f"Invalid tool name format: {tool_name}"
            )

        _, action = parts

        # Execute via MiniLab interface
        result = await self.tool.execute(action, arguments)

        # Convert to MCP format
        if result.success:
            return {
                "content": [
                    {"type": "text", "text": str(result.data) if result.data else ""}
                ]
            }
        else:
            return {
                "content": [
                    {"type": "text", "text": f"Error: {result.error}"}
                ],
                "isError": True,
            }


class MCPServer(ABC):
    """
    Base class for MCP servers exposing MiniLab tools.
    
    Implement transport-specific subclasses (stdio, HTTP, WebSocket).
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
    ):
        """
        Initialize MCP server.
        
        Args:
            name: Server name
            version: Server version
        """
        self.name = name
        self.version = version
        self._tools: dict[str, MCPToolAdapter] = {}
        self._initialized = False

    def register_tool(self, tool: Tool) -> None:
        """Register a MiniLab tool."""
        adapter = MCPToolAdapter(tool)
        self._tools[tool.name] = adapter

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle an incoming MCP request.
        
        Args:
            request: The MCP request
            
        Returns:
            MCP response
        """
        try:
            if request.method == MCPMethod.INITIALIZE:
                result = await self._handle_initialize(request.params or {})
            elif request.method == MCPMethod.TOOLS_LIST:
                result = await self._handle_tools_list()
            elif request.method == MCPMethod.TOOLS_CALL:
                result = await self._handle_tools_call(request.params or {})
            else:
                raise MCPError(
                    MCP_METHOD_NOT_FOUND,
                    f"Unknown method: {request.method}"
                )

            return MCPResponse(id=request.id, result=result)

        except MCPError as e:
            return MCPResponse(id=request.id, error=e.to_dict())
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error=MCPError(MCP_INTERNAL_ERROR, str(e)).to_dict()
            )

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
        }

    async def _handle_tools_list(self) -> dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for adapter in self._tools.values():
            tools.extend([s.model_dump() for s in adapter.to_mcp_schema()])
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise MCPError(MCP_INVALID_PARAMS, "Missing tool name")

        # Find the right adapter
        tool_base = name.split(".")[0]
        adapter = self._tools.get(tool_base)

        if not adapter:
            raise MCPError(MCP_METHOD_NOT_FOUND, f"Unknown tool: {name}")

        return await adapter.execute_mcp(name, arguments)

    @abstractmethod
    async def start(self) -> None:
        """Start the server."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass


class MCPClient(ABC):
    """
    Base class for MCP clients consuming external servers.
    
    Allows MiniLab to use tools from external MCP servers.
    """

    def __init__(self, server_name: str):
        """
        Initialize MCP client.
        
        Args:
            server_name: Name of the MCP server to connect to
        """
        self.server_name = server_name
        self._tools: list[MCPToolSchema] = []
        self._connected = False
        self._request_id = 0

    def _next_id(self) -> str:
        """Generate next request ID."""
        self._request_id += 1
        return f"req_{self._request_id}"

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send a request and wait for response."""
        pass

    async def initialize(self) -> dict[str, Any]:
        """Initialize the connection."""
        request = MCPRequest(
            id=self._next_id(),
            method=MCPMethod.INITIALIZE,
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MiniLab",
                    "version": "0.4.0",
                },
            },
        )
        response = await self.send_request(request)

        if response.error:
            raise MCPError(**response.error)

        self._connected = True
        return response.result or {}

    async def list_tools(self) -> list[MCPToolSchema]:
        """List available tools from the server."""
        request = MCPRequest(
            id=self._next_id(),
            method=MCPMethod.TOOLS_LIST,
        )
        response = await self.send_request(request)

        if response.error:
            raise MCPError(**response.error)

        tools_data = (response.result or {}).get("tools", [])
        self._tools = [MCPToolSchema.model_validate(t) for t in tools_data]
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        request = MCPRequest(
            id=self._next_id(),
            method=MCPMethod.TOOLS_CALL,
            params={
                "name": name,
                "arguments": arguments,
            },
        )
        response = await self.send_request(request)

        if response.error:
            raise MCPError(**response.error)

        return response.result or {}


class MCPToolBridge(Tool):
    """
    Bridge that exposes an MCP server's tools as MiniLab Tools.
    
    Allows external MCP tools to be used within MiniLab's tool system.
    """

    name = "mcp_bridge"
    description = "Bridge to external MCP tools"

    def __init__(
        self,
        agent_id: str,
        client: MCPClient,
        **kwargs: Any,
    ):
        """
        Initialize bridge.
        
        Args:
            agent_id: Agent ID
            client: Connected MCP client
        """
        super().__init__(agent_id, **kwargs)
        self.client = client
        self._tool_cache: dict[str, MCPToolSchema] = {}

    async def refresh_tools(self) -> None:
        """Refresh the list of available tools."""
        tools = await self.client.list_tools()
        self._tool_cache = {t.name: t for t in tools}

    def get_actions(self) -> dict[str, str]:
        """Return available actions (MCP tools)."""
        return {
            tool.name: tool.description
            for tool in self._tool_cache.values()
        }

    def get_input_schema(self, action: str) -> type[ToolInput]:
        """Get input schema for an MCP tool."""
        # Return a dynamic ToolInput - in practice you'd generate this
        # from the MCP tool's inputSchema
        return ToolInput

    async def execute(self, action: str, params: dict[str, Any]) -> ToolOutput:
        """Execute an MCP tool."""
        try:
            result = await self.client.call_tool(action, params)

            # Extract text content
            content = result.get("content", [])
            text_parts = [
                c.get("text", "")
                for c in content
                if c.get("type") == "text"
            ]

            is_error = result.get("isError", False)

            return ToolOutput(
                success=not is_error,
                data="\n".join(text_parts),
                error=text_parts[0] if is_error and text_parts else None,
            )

        except MCPError as e:
            return ToolOutput(success=False, error=str(e))
        except Exception as e:
            return ToolOutput(success=False, error=f"MCP call failed: {e}")
