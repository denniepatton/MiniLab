#!/usr/bin/env python
"""
Daily Literature Digest Script

Generates personalized paper recommendations based on project topics
and connections to existing work.
"""

import asyncio
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab.storage.state_store import StateStore
from MiniLab.bibliography import DailyDigest


async def main():
    # Initialize state store
    store = StateStore()
    
    # List available projects
    projects = store.list_projects()
    
    if not projects:
        print("No projects found. Create a project first with manage_project.py")
        return
    
    print("Available projects:")
    for i, proj in enumerate(projects, 1):
        print(f"{i}. {proj['name']} (ID: {proj['project_id']})")
    
    # Get project selection
    choice = input("\nSelect project number (or press Enter for general digest): ").strip()
    
    project_id = None
    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(projects):
            project_id = projects[idx]["project_id"]
            print(f"\nGenerating digest for project: {projects[idx]['name']}")
    
    # Configure topics
    print("\nEnter research topics (comma-separated):")
    topics_input = input("> ").strip()
    
    if topics_input:
        topics = [t.strip() for t in topics_input.split(",")]
    else:
        # Default topics for genomics/ML research
        topics = [
            "deep learning cancer genomics",
            "single-cell RNA-seq analysis",
            "causal inference genetics",
        ]
    
    print(f"\nScanning literature for topics: {', '.join(topics)}")
    print("This may take a moment...\n")
    
    # Generate recommendations
    digest = DailyDigest(state_store=store, topics=topics)
    
    try:
        recommendations = await digest.generate_daily_recommendations(
            project_id=project_id,
            num_papers=3,
        )
        
        # Format and display
        email = digest.format_recommendation_email(recommendations)
        print(email)
        
        # Optionally save to file
        save = input("\nSave to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"digest_{recommendations['date'][:10]}.txt"
            with open(filename, 'w') as f:
                f.write(email)
            print(f"Saved to {filename}")
        
    except Exception as e:
        print(f"Error generating digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
