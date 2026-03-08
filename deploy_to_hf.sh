#!/usr/bin/env bash
# =============================================================================
# MarketForge - HuggingFace Spaces Deployment Reference
# Space: https://huggingface.co/spaces/kenmandal/market-forge-env
# User:  kenmandal  |  Token: openenv-token
# =============================================================================
# These are the exact commands used to deploy successfully.
# Run from inside the market-forge-openenv/ directory.
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: Authenticate with HuggingFace
#         (Already stored as 'openenv-token' -- only re-run if token expires)
# -----------------------------------------------------------------------------
huggingface-cli login
# Paste your write-access token from: https://huggingface.co/settings/tokens

# -----------------------------------------------------------------------------
# STEP 2: Deploy (upload all files to the Space)
# -----------------------------------------------------------------------------
python3 - <<'EOF'
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path=".",                        # deploy from current directory
    repo_id="kenmandal/market-forge-env",   # HF Space repo
    repo_type="space",
    ignore_patterns=[
        "__pycache__",
        "*.pyc",
        "*.pptx",
        "*.docx",
        "*.zip",
        "tests/",
        "training_results.png",
        "deploy_to_hf.sh",
        "*_1.py",
        "~$*",
    ],
    commit_message="Deploy Multi-Agent MarketForge",
)

print("Deployed to: https://huggingface.co/spaces/kenmandal/market-forge-env")
EOF

# -----------------------------------------------------------------------------
# Space is live at:
#   https://huggingface.co/spaces/kenmandal/market-forge-env
#
# Files deployed:
#   README.md, Dockerfile, requirements.txt
#   app_visual.py, client.py, models.py, rewards.py, __init__.py
#   server/__init__.py, server/app.py, server/market_environment.py
#   train_market_forge.py, train_market_forge_notebook.py
#
# Connect an agent to the live Space:
#   from client import MarketForgeEnv
#   env = MarketForgeEnv(base_url="https://kenmandal-market-forge-env.hf.space")
# -----------------------------------------------------------------------------
