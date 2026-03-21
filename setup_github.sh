#!/bin/bash
# GitHub Setup Script for DocAssist

set -e

echo "DocAssist GitHub Setup Script"
echo "============================="
echo ""

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI is not authenticated."
    echo "Please run: gh auth login"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

REPO_URL="https://github.com/sweeden-ttu/DocAssist"
REPO_DIR="/home/sweeden/projects/DocAssist"

cd "$REPO_DIR"

# Initialize git if not already
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git remote add origin "$REPO_URL"
else
    echo "Git repository already initialized"
fi

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.pytest_cache/
.coverage
htmlcov/
*.log
output/
models/
*.png
!.gitkeep
EOF

# Add files
echo "Adding files to git..."
git add .

# Commit
echo "Creating initial commit..."
git commit -m "feat: Initial DocAssist project structure

- Dual-model ensemble system (CPU + MLX)
- Form field detection with bounding boxes
- Docling MCP integration
- PyQt GUI viewer
- Episodic training pipeline
- IRS form support

Architecture:
- Validator: Qwen2.5-VL on Linux CPU (localhost:1234)
- Trainer: Qwen2.5-VL on Mac MLX (192.168.0.13:1234)"

# Create GitHub repository
echo "Creating GitHub repository..."
if gh repo create sweeden-ttu/DocAssist --public --source=. --push 2>/dev/null; then
    echo "Repository created and pushed!"
else
    echo "Repository may already exist. Pushing to existing..."
    git push -u origin main || git push -u origin master
fi

# Create GitHub Project
echo ""
echo "Creating GitHub Project..."
PROJECT_NAME="DocAssist Development"

# Check if gh CLI has project extension
if gh extension list | grep -q projects; then
    gh project create "$PROJECT_NAME" --owner sweeden-ttu --template "Blank"
else
    echo "GitHub Projects CLI not installed. Install with: gh extension install gh-projects"
fi

echo ""
echo "Setup complete!"
echo "Repository: https://github.com/sweeden-ttu/DocAssist"
echo ""
echo "Next steps:"
echo "1. Create GitHub Project at: https://github.com/users/sweeden-ttu/projects/new"
echo "2. Add issues for each milestone"
echo "3. Enable LM Link on Mac for remote model access"
echo "4. Download models:"
echo "   - Local: lms get Qwen/Qwen2.5-VL-7B-Instruct-GGUF"
echo "   - Mac: lms get lmstudio-community/Qwen2.5-VL-7B-Instruct-4bit-MLX"
