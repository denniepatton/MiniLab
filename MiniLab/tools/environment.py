from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, List
from pathlib import Path

from . import Tool


class EnvironmentTool(Tool):
    """
    Manage Python packages and environment installations.
    Restricted to the minilab micromamba environment for safety.
    
    CRITICAL: ALL package installations require user permission.
    The system will pause and ask the user before installing anything.
    """

    def __init__(self, environment_name: str = "minilab", permission_callback=None):
        super().__init__(
            name="environment",
            description=(
                "Install Python packages and manage the conda environment. "
                "IMPORTANT: All installations require user approval - the system "
                "will pause and ask permission before installing any package."
            )
        )
        self.environment_name = environment_name
        self.permission_callback = permission_callback  # Async function to ask user permission
        
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute environment management operation.
        
        Actions:
        - install_package: Install Python package(s) (requires: packages [list])
        - install_tool: Install system tool via conda/brew (requires: tool, method)
        - list_packages: List installed packages
        - check_package: Check if package is installed (requires: package)
        """
        try:
            if action == "install_package":
                return await self._install_packages(kwargs.get("packages", []))
            elif action == "install_tool":
                return await self._install_tool(
                    kwargs["tool"], 
                    kwargs.get("method", "conda")
                )
            elif action == "list_packages":
                return await self._list_packages()
            elif action == "check_package":
                return await self._check_package(kwargs["package"])
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "install_package", "install_tool", 
                        "list_packages", "check_package"
                    ]
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action,
            }
    
    async def _install_packages(self, packages: List[str]) -> Dict[str, Any]:
        """
        Install Python package(s) via pip in the minilab environment.
        
        ALWAYS requires user permission before installing.
        
        Args:
            packages: List of package names (can include version specs like "pandas>=2.0")
        """
        if not packages:
            return {
                "success": False,
                "error": "No packages specified",
            }
        
        # Normalize package names for display
        package_names = []
        for pkg in packages:
            # Extract base package name (before any version specifiers)
            base_name = pkg.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip()
            package_names.append(base_name)
        
        # ALWAYS ask for permission - no automatic approval
        if self.permission_callback:
            permission_granted = await self.permission_callback(
                f"install Python packages: {', '.join(packages)}"
            )
            if not permission_granted:
                return {
                    "success": False,
                    "error": f"Permission denied to install: {', '.join(packages)}",
                    "permission_required": True,
                    "message": "User declined the installation request.",
                }
        else:
            # No callback configured - cannot proceed without user approval mechanism
            return {
                "success": False,
                "error": "Cannot install packages: no user approval mechanism configured",
                "permission_required": True,
                "packages": packages,
                "message": (
                    "Package installation requires user approval. "
                    "Please configure a permission_callback or manually approve this installation."
                ),
            }
        
        # Install packages (only if permission granted)
        try:
            # Use micromamba run to execute pip in the minilab environment
            cmd = [
                "micromamba", "run", "-n", self.environment_name,
                "pip", "install"
            ] + packages
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "action": "install_package",
                    "packages": packages,
                    "stdout": result.stdout,
                    "message": f"Successfully installed: {', '.join(packages)}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Installation failed: {result.stderr}",
                    "stdout": result.stdout,
                    "return_code": result.returncode,
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Installation timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Installation error: {str(e)}",
            }
    
    async def _install_tool(self, tool: str, method: str = "conda") -> Dict[str, Any]:
        """
        Install a system tool (like R, bedtools, etc.) via conda or brew.
        Always requires permission.
        
        Args:
            tool: Tool name (e.g., "r-base", "bedtools")
            method: Installation method ("conda" or "brew")
        """
        # Always ask permission for system tools
        if self.permission_callback:
            permission_granted = await self.permission_callback(
                f"install system tool '{tool}' via {method}"
            )
            if not permission_granted:
                return {
                    "success": False,
                    "error": f"Permission denied to install {tool}",
                    "permission_required": True,
                }
        
        try:
            if method == "conda":
                cmd = [
                    "micromamba", "install", "-n", self.environment_name,
                    "-c", "conda-forge", "-y", tool
                ]
            elif method == "brew":
                cmd = ["brew", "install", tool]
            else:
                return {
                    "success": False,
                    "error": f"Unknown installation method: {method}. Use 'conda' or 'brew'",
                }
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large tools
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "action": "install_tool",
                    "tool": tool,
                    "method": method,
                    "message": f"Successfully installed {tool} via {method}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Installation failed: {result.stderr}",
                    "return_code": result.returncode,
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Installation error: {str(e)}",
            }
    
    async def _list_packages(self) -> Dict[str, Any]:
        """List all installed packages in the minilab environment."""
        try:
            result = subprocess.run(
                ["micromamba", "run", "-n", self.environment_name, "pip", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "packages": result.stdout,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _check_package(self, package: str) -> Dict[str, Any]:
        """Check if a package is installed."""
        try:
            result = subprocess.run(
                ["micromamba", "run", "-n", self.environment_name, 
                 "python", "-c", f"import {package}"],
                capture_output=True,
                text=True,
                timeout=30  # Increased from 10 to 30 seconds
            )
            
            if result.returncode == 0:
                # Try to get version
                version_result = subprocess.run(
                    ["micromamba", "run", "-n", self.environment_name,
                     "python", "-c", f"import {package}; print({package}.__version__)"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                
                return {
                    "success": True,
                    "installed": True,
                    "package": package,
                    "version": version,
                }
            else:
                return {
                    "success": True,
                    "installed": False,
                    "package": package,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
