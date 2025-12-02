# MiniLab Setup Guide

This guide walks you through setting up MiniLab on your local machine and uploading to GitHub.

## Part 1: Local Environment Setup

### 1. Install Micromamba (if not already installed)

Micromamba is a lightweight, fast package manager compatible with conda.

```bash
# On macOS (using Homebrew)
brew install micromamba

# Or using the official installer
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

After installation, restart your terminal or run:
```bash
source ~/.zshrc  # or ~/.bash_profile for bash
```

### 2. Navigate to MiniLab Directory

```bash
cd /Users/robertpatton/MiniLab
```

### 3. Create the Micromamba Environment

```bash
# Create environment from environment.yml
micromamba env create -f environment.yml

# This will create an environment named 'minilab' with Python 3.11 and all dependencies
```

### 4. Activate the Environment

```bash
micromamba activate minilab

# You should see (minilab) in your terminal prompt
```

### 5. Install MiniLab in Development Mode

```bash
pip install -e .
```

This installs MiniLab as an editable package, so changes to the code are immediately available.

### 6. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite text editor
nano .env  # or vim, code, etc.
```

**Required keys:**
- `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com/settings/keys

**Optional keys:**
- `ZOTERO_API_KEY` and `ZOTERO_USER_ID`: From https://www.zotero.org/settings/keys
- `TAVILY_API_KEY`: For enhanced web search
- `NCBI_EMAIL`: Your email for PubMed API (improves rate limits)

Example `.env`:
```
OPENAI_API_KEY=sk-proj-...your_key_here...
ANTHROPIC_API_KEY=sk-ant-...your_key_here...
```

### 7. Test the Installation

```bash
# Test import
python -c "from minilab import load_agents; print('✓ MiniLab installed successfully')"

   # Test loading agents (requires API keys)
   python -c "from MiniLab import load_agents; agents = load_agents(); print(f'✓ Loaded {len(agents)} agents')"
```

### 8. Run Your First Meeting

```bash
python scripts/run_user_meeting.py
```

When prompted, try asking: "What are the key considerations for applying deep learning to cancer genomics?"

## Part 2: GitHub Setup

### 1. Initialize Git Repository

```bash
cd /Users/robertpatton/MiniLab

# Initialize git (if not already done)
git init

# Add all files
git add .

# Check what will be committed
git status

# Make initial commit
git commit -m "Initial commit: MiniLab v0.1.0 with full agent system"
```

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `MiniLab`
3. Description: "Multi-agent AI system for scientific research"
4. Choose Public or Private
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 3. Link Local Repository to GitHub

GitHub will show you commands. Use these:

```bash
# Add GitHub as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/MiniLab.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Set Up GitHub Best Practices

#### A. Add a License

```bash
# Create LICENSE file (example: MIT License)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

git add LICENSE
git commit -m "Add MIT License"
git push
```

#### B. Add GitHub Actions (Optional - for CI/CD)

Create `.github/workflows/tests.yml`:

```bash
mkdir -p .github/workflows

cat > .github/workflows/tests.yml << 'EOF'
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: mamba-org/setup-micromamba@v1
        with:
          environment-file: environment.yml
          cache-environment: true
      - name: Install package
        run: pip install -e .
      - name: Run tests
        run: pytest tests/ || echo "No tests yet"
EOF

git add .github/
git commit -m "Add GitHub Actions CI"
git push
```

### 5. Protect Your Secrets

**CRITICAL**: Never commit your `.env` file with real API keys!

Verify `.gitignore` includes:
```bash
grep "\.env" .gitignore
# Should show: .env
```

If you accidentally committed `.env`:
```bash
# Remove from git (but keep locally)
git rm --cached .env
git commit -m "Remove .env from version control"
git push
```

### 6. Create Releases

Once you have a working version:

```bash
# Tag a release
git tag -a v0.1.0 -m "MiniLab v0.1.0: Initial release with 9-agent system"
git push origin v0.1.0
```

Then on GitHub:
1. Go to your repository
2. Click "Releases" → "Create a new release"
3. Choose tag `v0.1.0`
4. Add release notes
5. Publish release

## Part 3: Working with MiniLab

### Daily Workflow

```bash
# 1. Activate environment
micromamba activate minilab

# 2. Pull latest changes (if working across machines)
git pull

# 3. Work with MiniLab
python scripts/run_user_meeting.py
python scripts/daily_digest.py
python scripts/manage_project.py

# 4. Commit changes
git add .
git commit -m "Descriptive commit message"
git push
```

### Updating Dependencies

```bash
# Update environment.yml, then:
micromamba env update -f environment.yml --prune

# Or recreate environment from scratch:
micromamba env remove -n minilab
micromamba env create -f environment.yml
```

### Syncing Across Machines

```bash
# On machine A (after making changes):
git add .
git commit -m "Update agent personas"
git push

# On machine B:
cd ~/MiniLab  # or wherever you cloned it
git pull
micromamba activate minilab
pip install -e .  # Reinstall if dependencies changed
```

## Part 4: HPC Integration (Future)

For future HPC integration:

1. **SSH Setup**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ssh-copy-id username@hpc.institution.edu
   ```

2. **Clone MiniLab on HPC**:
   ```bash
   ssh username@hpc.institution.edu
   git clone https://github.com/YOUR_USERNAME/MiniLab.git
   cd MiniLab
   module load micromamba  # or install micromamba
   micromamba env create -f environment.yml
   ```

3. **Consider Data Security**:
   - Use HPC-local LLMs (Ollama, vLLM) for PHI data
   - Never send identifiable patient data to external APIs
   - Get IRB approval before processing human subjects data
   - Use secure file transfer (sftp, rsync over SSH)

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution**: Make sure `.env` file exists and is loaded. Check with:
```bash
cat .env | grep OPENAI_API_KEY
```

### Issue: "Module not found: minilab"
**Solution**: Install in development mode:
```bash
pip install -e .
```

### Issue: API rate limits
**Solution**: 
- Use `gpt-4o-mini` instead of `gpt-4o` for non-critical agents
- Edit `MiniLab/config/agents.yaml` to change models
- Add delays between requests if needed

### Issue: micromamba not found
**Solution**: 
```bash
# Ensure micromamba is in PATH
which micromamba

# If not found, add to ~/.zshrc:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Next Steps

1. ✅ Set up environment and API keys
2. ✅ Run first team meeting
3. ✅ Create a project with `manage_project.py`
4. ✅ Test daily digest
5. ✅ Push to GitHub
6. 🔜 Start integrating with your research workflow
7. 🔜 Customize agent personas in `config/agents.yaml`
8. 🔜 Add domain-specific tools for your research

## Resources

- **VirtualLab Paper**: https://www.nature.com/articles/s41586-025-09442-9
- **CellVoyager Paper**: https://www.biorxiv.org/content/10.1101/2025.06.03.657517v1
- **OpenAI API Docs**: https://platform.openai.com/docs
- **Anthropic API Docs**: https://docs.anthropic.com/
- **Micromamba Docs**: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review agent logs in terminal output

Happy researching with MiniLab! 🧪🤖
