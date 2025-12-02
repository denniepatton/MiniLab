"""
Example: Test FileSystem Tool with Sandbox

This demonstrates how agents can safely interact with files in the Sandbox.
"""

import asyncio
from pathlib import Path

from MiniLab import load_agents


async def main():
    agents = load_agents()
    lee = agents["lee"]
    
    print(f"\n{lee.display_name} has access to: {list(lee.tool_instances.keys())}")
    print(f"Sandbox root: {Path(__file__).parent.parent / 'Sandbox'}")
    
    # Test 1: Create a directory
    print("\n" + "=" * 60)
    print("Test 1: Create directory 'projects/test'")
    print("=" * 60)
    result = await lee.use_tool("filesystem", "create_dir", path="projects/test")
    print(result)
    
    # Test 2: Write a file
    print("\n" + "=" * 60)
    print("Test 2: Write a Python script")
    print("=" * 60)
    result = await lee.use_tool(
        "filesystem", 
        "write",
        path="projects/test/hello.py",
        content='print("Hello from MiniLab!")\n'
    )
    print(result)
    
    # Test 3: Read the file back
    print("\n" + "=" * 60)
    print("Test 3: Read the file back")
    print("=" * 60)
    result = await lee.use_tool("filesystem", "read", path="projects/test/hello.py")
    print(f"Success: {result['success']}")
    print(f"Content:\n{result['content']}")
    
    # Test 4: List directory
    print("\n" + "=" * 60)
    print("Test 4: List 'projects/test'")
    print("=" * 60)
    result = await lee.use_tool("filesystem", "list", path="projects/test")
    print(result)
    
    # Test 5: Try to escape sandbox (should fail)
    print("\n" + "=" * 60)
    print("Test 5: Attempt to access parent directory (should fail)")
    print("=" * 60)
    result = await lee.use_tool("filesystem", "read", path="../pyproject.toml")
    print(result)
    
    print("\n✓ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
