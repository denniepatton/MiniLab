"""
Example: Agent using filesystem tool in conversation

Demonstrates how an agent (Lee) can be asked to create files via natural conversation.
"""

import asyncio
from pathlib import Path

from MiniLab import load_agents


async def main():
    agents = load_agents()
    lee = agents["lee"]
    
    print("\n" + "=" * 60)
    print("Example: Asking Lee to create a Python analysis script")
    print("=" * 60 + "\n")
    
    # In a real conversation, you'd ask Franklin who would delegate to Lee
    # Here we're asking Lee directly for demonstration
    
    request = """
    Please create a Python script in Sandbox/projects/ called data_loader.py
    that demonstrates loading a CSV file with pandas. Include proper error handling
    and type hints.
    """
    
    print(f"User request:\n{request}\n")
    
    # Lee responds (but doesn't actually execute the tool yet - that's Phase 2)
    response = await lee.arespond(request)
    print(f"{lee.display_name}'s response:\n{response}\n")
    
    # Now we manually execute the tool based on Lee's plan
    # (In Phase 2, Lee will be able to invoke tools directly)
    print("=" * 60)
    print("Executing filesystem operations based on Lee's plan...")
    print("=" * 60 + "\n")
    
    script_content = '''"""
Data loader utility for MiniLab analysis projects.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(
    filepath: str | Path,
    *,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV file with error handling and validation.
    
    Args:
        filepath: Path to the CSV file
        required_columns: Optional list of columns that must be present
        
    Returns:
        pandas DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Validate required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return df


if __name__ == "__main__":
    # Example usage
    try:
        df = load_csv("data/example.csv", required_columns=["id", "value"])
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
'''
    
    # Create the directory
    result = await lee.use_tool("filesystem", "create_dir", path="projects")
    print(f"Created directory: {result}")
    
    # Write the file
    result = await lee.use_tool(
        "filesystem",
        "write",
        path="projects/data_loader.py",
        content=script_content
    )
    print(f"\nCreated file: {result}")
    
    # Verify it exists
    result = await lee.use_tool("filesystem", "list", path="projects")
    print(f"\nDirectory contents: {result}")
    
    print("\n✓ File created successfully!")
    print(f"✓ Location: {Path.cwd()}/Sandbox/projects/data_loader.py")


if __name__ == "__main__":
    asyncio.run(main())
