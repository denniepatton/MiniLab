"""
Environment Tool with PathGuard integration.

Provides package/tool installation with permission controls.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..security import PathGuard


class InstallPackageInput(ToolInput):
    """Input for installing a Python package."""
    package: str = Field(..., description="Package name (optionally with version, e.g., 'pandas>=2.0')")
    upgrade: bool = Field(False, description="Whether to upgrade if already installed")


class InstallToolInput(ToolInput):
    """Input for installing a system tool."""
    tool: str = Field(..., description="Tool name")
    method: str = Field("conda", description="Installation method: 'conda', 'brew', 'apt'")


class ListPackagesInput(ToolInput):
    """Input for listing installed packages."""
    filter: Optional[str] = Field(None, description="Filter packages by name substring")


class CheckPackageInput(ToolInput):
    """Input for checking if a package is installed."""
    package: str = Field(..., description="Package name to check")


class EnvironmentOutput(ToolOutput):
    """Output for environment operations."""
    package: Optional[str] = None
    version: Optional[str] = None
    packages: Optional[list[dict]] = None
    installed: Optional[bool] = None


class EnvironmentTool(Tool):
    """
    Environment management with permission controls.
    
    Package installations require explicit user permission via callback.
    Only Bohr (and agents with can_modify_environment) can approve changes.
    """
    
    name = "environment"
    description = "Manage Python packages and system tools (requires user permission)"
    
    def __init__(self, agent_id: str, workspace_root: Path, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.workspace_root = workspace_root
        self.path_guard = PathGuard.get_instance()
    
    def get_actions(self) -> dict[str, str]:
        return {
            "install_package": "Install a Python package",
            "install_tool": "Install a system tool",
            "list_packages": "List installed Python packages",
            "check_package": "Check if a package is installed",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "install_package": InstallPackageInput,
            "install_tool": InstallToolInput,
            "list_packages": ListPackagesInput,
            "check_package": CheckPackageInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> EnvironmentOutput:
        """Execute an environment action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "install_package":
                return await self._install_package(validated)
            elif action == "install_tool":
                return await self._install_tool(validated)
            elif action == "list_packages":
                return await self._list_packages(validated)
            elif action == "check_package":
                return await self._check_package(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return EnvironmentOutput(success=False, error=f"Operation failed: {e}")
    
    async def _request_permission(self, description: str) -> bool:
        """Request permission for environment modification."""
        if self.permission_callback:
            return self.permission_callback(description)
        
        # If no callback, only allow agents with permission
        return self.path_guard.can_modify_environment(self.agent_id)
    
    async def _install_package(self, params: InstallPackageInput) -> EnvironmentOutput:
        """Install a Python package."""
        permission_msg = f"Install Python package: {params.package}"
        if not await self._request_permission(permission_msg):
            return EnvironmentOutput(
                success=False,
                package=params.package,
                error="Permission denied for package installation"
            )
        
        cmd = [sys.executable, "-m", "pip", "install"]
        if params.upgrade:
            cmd.append("--upgrade")
        cmd.append(params.package)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                # Try to get installed version
                version = None
                try:
                    import importlib.metadata
                    pkg_name = params.package.split("[")[0].split(">=")[0].split("==")[0].split("<")[0]
                    version = importlib.metadata.version(pkg_name)
                except Exception:
                    pass
                
                return EnvironmentOutput(
                    success=True,
                    package=params.package,
                    version=version,
                    data=f"Successfully installed {params.package}"
                )
            else:
                return EnvironmentOutput(
                    success=False,
                    package=params.package,
                    error=f"Installation failed: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            return EnvironmentOutput(
                success=False,
                package=params.package,
                error="Installation timed out"
            )
    
    async def _install_tool(self, params: InstallToolInput) -> EnvironmentOutput:
        """Install a system tool."""
        permission_msg = f"Install system tool: {params.tool} via {params.method}"
        if not await self._request_permission(permission_msg):
            return EnvironmentOutput(
                success=False,
                error="Permission denied for tool installation"
            )
        
        if params.method == "conda":
            cmd = ["conda", "install", "-y", params.tool]
        elif params.method == "brew":
            cmd = ["brew", "install", params.tool]
        elif params.method == "apt":
            cmd = ["sudo", "apt-get", "install", "-y", params.tool]
        else:
            return EnvironmentOutput(
                success=False,
                error=f"Unknown installation method: {params.method}"
            )
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if result.returncode == 0:
                return EnvironmentOutput(
                    success=True,
                    data=f"Successfully installed {params.tool}"
                )
            else:
                return EnvironmentOutput(
                    success=False,
                    error=f"Installation failed: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            return EnvironmentOutput(
                success=False,
                error="Installation timed out"
            )
        except FileNotFoundError:
            return EnvironmentOutput(
                success=False,
                error=f"{params.method} not found. Please install it first."
            )
    
    async def _list_packages(self, params: ListPackagesInput) -> EnvironmentOutput:
        """List installed Python packages."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                import json
                packages = json.loads(result.stdout)
                
                if params.filter:
                    packages = [p for p in packages if params.filter.lower() in p["name"].lower()]
                
                return EnvironmentOutput(
                    success=True,
                    packages=packages,
                    data=f"Found {len(packages)} packages"
                )
            else:
                return EnvironmentOutput(
                    success=False,
                    error=f"Failed to list packages: {result.stderr}"
                )
        except Exception as e:
            return EnvironmentOutput(success=False, error=str(e))
    
    async def _check_package(self, params: CheckPackageInput) -> EnvironmentOutput:
        """Check if a package is installed."""
        try:
            import importlib.metadata
            version = importlib.metadata.version(params.package)
            return EnvironmentOutput(
                success=True,
                package=params.package,
                version=version,
                installed=True,
            )
        except importlib.metadata.PackageNotFoundError:
            return EnvironmentOutput(
                success=True,
                package=params.package,
                installed=False,
            )
        except Exception as e:
            return EnvironmentOutput(success=False, error=str(e))
