#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_github.sh [repo-slug] [public|private]
# Example: ./setup_github.sh entropic-collatz-conjector public

# 1. Parse arguments
REPO_SLUG=${1:-entropic-collatz-conjector}
VISIBILITY=${2:-public}

# 2. Full human-readable title for README
FULL_TITLE="Entropic Collatz Conjector"

echo "🔧 Scaffolding project for GitHub as '$REPO_SLUG' ($VISIBILITY)…"

# 3. Create necessary directories
mkdir -p src configs tests .github/workflows

# 4. Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.env
venv/

# Logs
*.log

# macOS files
.DS_Store

# Windows files
Thumbs.db

# IDE / Editor folders
.vscode/
.idea/
EOF

# 5. Create requirements.txt
pip freeze > requirements.txt

# 6. Create README.md
cat > README.md << EOF
# $FULL_TITLE

_Repository slug:_ \`$REPO_SLUG\`

---

## Abstract
This project explores the Collatz Conjecture using entropy and clustering techniques. It involves analyzing the behavior of the Collatz sequence and its related properties.

---

## Installation

\`\`\`bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
\`\`\`

---

## Usage

\`\`\`bash
python entropic_collatz_conjector.py
\`\`\`

---

## Project Structure

$REPO_SLUG/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── entropic_collatz_conjector.py
├── test_seed.py
├── src/
│ └── your_module.py
├── configs/
│ └── default.yaml
├── tests/
│ └── test_basic.py
└── .github/
└── workflows/
└── ci.yml

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
EOF

# 7. Create LICENSE (MIT)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
EOF

# 8. Create src/your_module.py
cat > src/your_module.py << 'EOF'
def core_function(x):
    """
    TODO: Implement the core functionality.
    """
    return 42
EOF

# 9. Create configs/default.yaml
cat > configs/default.yaml << 'EOF'
param1: 0.1
param2: 100
EOF

# 10. Create tests/test_basic.py
cat > tests/test_basic.py << 'EOF'
import pytest
from src.your_module import core_function

def test_core_function_returns_expected():
    assert core_function(0) == 42
EOF

# 11. Create GitHub Actions CI workflow
cat > .github/workflows/ci.yml << 'EOF'
name: Continuous Integration

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings -q
EOF

# 12. Initialize Git and make the first commit
if [ ! -d .git ]; then
  git init
fi
git add .
git commit -m "chore: initial scaffold with docs, tests, and CI workflow"

# 13. Create GitHub repo & push (if gh CLI is available)
if command -v gh &> /dev/null; then
  echo "🔗 Creating GitHub repository '$REPO_SLUG' ($VISIBILITY)…"
  gh repo create "$REPO_SLUG" --"$VISIBILITY" --source=. --remote=origin --push
else
  echo "⚠️  GitHub CLI not found. Please manually create a repo named '$REPO_SLUG', then:"
  echo "    git remote add origin git@github.com:YOUR_USERNAME/$REPO_SLUG.git"
  echo "    git branch -M main"
  echo "    git push -u origin main"
fi

echo "✅ Project is now GitHub-ready!"