#!/usr/bin/env python
"""
Project Management Script

Create and manage MiniLab research projects, including:
- Creating new projects
- Adding citations
- Building knowledge graphs
- Viewing project status
"""

import asyncio
import sys
from pathlib import Path
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from MiniLab.storage.state_store import StateStore, Citation, ConceptLink


def print_menu():
    print("\n" + "=" * 60)
    print("MiniLab Project Manager")
    print("=" * 60)
    print("1. Create new project")
    print("2. List projects")
    print("3. View project details")
    print("4. Add citation to project")
    print("5. Add concept link")
    print("6. Export bibliography")
    print("7. Search citations")
    print("8. Exit")
    print("=" * 60)


def create_project(store: StateStore):
    print("\n--- Create New Project ---")
    project_id = input("Project ID (e.g., 'my_research'): ").strip()
    name = input("Project name: ").strip()
    description = input("Project description: ").strip()
    
    if not all([project_id, name, description]):
        print("Error: All fields required")
        return
    
    try:
        project = store.create_project(project_id, name, description)
        print(f"✓ Created project '{name}' (ID: {project_id})")
    except Exception as e:
        print(f"Error creating project: {e}")


def list_projects(store: StateStore):
    projects = store.list_projects()
    
    if not projects:
        print("\nNo projects found.")
        return
    
    print(f"\n--- Projects ({len(projects)}) ---")
    for proj in projects:
        print(f"\nID: {proj['project_id']}")
        print(f"Name: {proj['name']}")
        print(f"Description: {proj['description']}")
        print(f"Last modified: {proj['last_modified']}")


def view_project_details(store: StateStore):
    project_id = input("\nProject ID: ").strip()
    project = store.load_project(project_id)
    
    if not project:
        print(f"Project '{project_id}' not found")
        return
    
    print(f"\n--- Project: {project.name} ---")
    print(f"Description: {project.description}")
    print(f"Created: {project.created_date}")
    print(f"Last modified: {project.last_modified}")
    print(f"\nCitations: {len(project.citations)}")
    print(f"Concept links: {len(project.concept_links)}")
    print(f"Ideas: {len(project.ideas)}")
    print(f"Meetings: {len(project.meeting_history)}")
    
    if project.citations:
        print("\nRecent citations:")
        for i, (key, cit) in enumerate(list(project.citations.items())[:5], 1):
            print(f"  {i}. [{cit.key}] {cit.title} ({cit.year})")


def add_citation(store: StateStore):
    project_id = input("\nProject ID: ").strip()
    project = store.load_project(project_id)
    
    if not project:
        print(f"Project '{project_id}' not found")
        return
    
    print("\n--- Add Citation ---")
    key = input("Citation key (e.g., 'Smith2020'): ").strip()
    title = input("Title: ").strip()
    authors_str = input("Authors (comma-separated): ").strip()
    authors = [a.strip() for a in authors_str.split(",")]
    year = int(input("Year: ").strip())
    journal = input("Journal (optional): ").strip() or None
    doi = input("DOI (optional): ").strip() or None
    
    citation = Citation(
        key=key,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        doi=doi,
    )
    
    project.add_citation(citation)
    store.save_project(project)
    print(f"✓ Added citation '{key}' to project")


def add_concept_link(store: StateStore):
    project_id = input("\nProject ID: ").strip()
    project = store.load_project(project_id)
    
    if not project:
        print(f"Project '{project_id}' not found")
        return
    
    print("\n--- Add Concept Link ---")
    source = input("Source (citation key or concept): ").strip()
    target = input("Target (citation key or concept): ").strip()
    relation = input("Relation type (e.g., 'cites', 'extends', 'contradicts'): ").strip()
    description = input("Description: ").strip()
    
    link = ConceptLink(
        source=source,
        target=target,
        relation_type=relation,
        description=description,
    )
    
    project.add_concept_link(link)
    store.save_project(project)
    print(f"✓ Added link: {source} --[{relation}]--> {target}")


def export_bibliography(store: StateStore):
    project_id = input("\nProject ID: ").strip()
    format_choice = input("Format (bibtex/json): ").strip().lower()
    
    if format_choice not in ["bibtex", "json"]:
        print("Invalid format")
        return
    
    try:
        output = store.export_bibliography(project_id, format=format_choice)
        
        if not output:
            print("No bibliography to export or project not found")
            return
        
        filename = f"{project_id}_bibliography.{format_choice}"
        with open(filename, 'w') as f:
            f.write(output)
        
        print(f"✓ Exported to {filename}")
        
    except Exception as e:
        print(f"Error exporting: {e}")


def search_citations(store: StateStore):
    query = input("\nSearch query: ").strip()
    project_id_input = input("Search in specific project? (press Enter for global): ").strip()
    
    project_id = project_id_input if project_id_input else None
    
    results = store.search_citations(query, project_id)
    
    if not results:
        print("No citations found")
        return
    
    print(f"\n--- Found {len(results)} citations ---")
    for i, cit in enumerate(results[:10], 1):
        print(f"\n{i}. [{cit.key}] {cit.title}")
        print(f"   Authors: {', '.join(cit.authors[:3])}")
        print(f"   Year: {cit.year}")
        if cit.doi:
            print(f"   DOI: {cit.doi}")


def main():
    store = StateStore()
    
    while True:
        print_menu()
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            create_project(store)
        elif choice == "2":
            list_projects(store)
        elif choice == "3":
            view_project_details(store)
        elif choice == "4":
            add_citation(store)
        elif choice == "5":
            add_concept_link(store)
        elif choice == "6":
            export_bibliography(store)
        elif choice == "7":
            search_citations(store)
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
