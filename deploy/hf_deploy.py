"""
Deploy the service to a Hugging Face Docker Space (free CPU tier, no card).

    HF_TOKEN=hf_xxx HF_SPACE=<user>/<space> python deploy/hf_deploy.py

Creates the Space if needed, copies GROQ/GEMINI/LANGFUSE keys from the environment
into Space secrets (only those that are set), and uploads just the files the Docker
build needs. Uses the Hub HTTP API, so binary index files need no git-lfs setup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
VARIABLES = ("GROQ_MODEL", "GROQ_FALLBACK_MODEL", "GEMINI_MODEL", "RATE_LIMIT", "DAILY_LLM_CAP", "RETRIEVAL_CONFIG")
UPLOAD = [
    "Dockerfile",
    ".dockerignore",
    "requirements-service.txt",
    "requirements-dense.txt",
    "platform_config.py",
    "config/**",
    "configs/**",
    "prompts/**",
    "retrieval/**",
    "evaluation/__init__.py",
    "evaluation/retrieval_metrics.py",
    "src/**",
    "data/index/chunks/**",
    "data/index/embeddings/*_bge.npy",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    space = os.environ.get("HF_SPACE", "").strip()
    if not token or not space:
        sys.exit("HF_TOKEN and HF_SPACE (e.g. yourname/rag-10k-api) are required")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    for k in SECRETS:
        if os.environ.get(k, "").strip():
            api.add_space_secret(space, k, os.environ[k].strip())
            print(f"secret set: {k}")
    for k in VARIABLES:
        if os.environ.get(k, "").strip():
            api.add_space_variable(space, k, os.environ[k].strip())
            print(f"variable set: {k}")

    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "hf_space_README.md"),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message="Update Space card",
    )
    sha = os.environ.get("GITHUB_SHA", "local")[:12]
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=space,
        repo_type="space",
        allow_patterns=UPLOAD,
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        delete_patterns=["src/**", "configs/**", "prompts/**", "retrieval/**"],
        commit_message=f"Deploy {sha}",
    )
    owner, name = space.split("/")
    print(f"deployed: https://huggingface.co/spaces/{space}")
    print(f"api:      https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
