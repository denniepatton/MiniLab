# Continuous Conversation & Filesystem Tool

## New Features (December 2025)

### 1. Continuous Conversation Mode

`run_user_meeting.py` now supports back-and-forth conversation:

```bash
python scripts/run_user_meeting.py
```

- Have multi-turn conversations with Franklin
- Conversation history maintained for context (last 3 exchanges)
- Type `exit`, `quit`, or `q` to end the session
- Session token usage tracked cumulatively

### 2. Sandbox Directory

Agents now have a safe workspace at `Sandbox/` for file operations:

- **Purpose**: Isolated directory where agents can create, edit, read, and write files
- **Security**: Path validation prevents access outside Sandbox
- **Git**: Structure tracked, contents ignored (see `.gitignore`)

### 3. FileSystem Tool

Lee (CS engineer) has access to the `filesystem` tool:

**Available Actions:**
- `read`: Read file contents
- `write`: Create or overwrite file
- `append`: Append to file
- `create_dir`: Create directory
- `list`: List directory contents
- `exists`: Check if path exists
- `delete`: Delete file or directory

**Usage Example:**
```python
# Within agent code or orchestrator
result = await agent.use_tool(
    "filesystem",
    "write",
    path="projects/analysis.py",
    content="import pandas as pd\n..."
)
```

**Test the tool:**
```bash
python scripts/test_filesystem_tool.py
```

### 4. Cost Controls

- Maximum 300,000 tokens per session
- Conservative estimate: ~2,000 tokens per agent response
- Default max_tokens=1000 per agent response (~700 words)
- Agents prompted to be concise

### Architecture Updates

**Agent.tool_instances**: Each agent now has instantiated tool objects
**Agent.use_tool()**: Method to execute tools safely
**Agent.has_tool()**: Check tool availability

**PI Coordination**: Franklin receives requests, consults specific team members with targeted questions, synthesizes responses

## Migration Notes

- Old `run_user_meeting.py` behavior (single question) replaced with continuous conversation
- All agents instructed to prioritize conciseness
- Filesystem operations restricted to Sandbox directory only

## Future Enhancements

- [ ] Tool result incorporation into LLM responses (currently tools are available but not auto-invoked)
- [ ] Web search tool integration
- [ ] Zotero citation tool
- [ ] Terminal execution tool with safety guards
- [ ] Multi-modal file handling (images, PDFs)
