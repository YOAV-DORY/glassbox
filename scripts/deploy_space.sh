#!/usr/bin/env bash
# Publish the synthetic-data demo to a Hugging Face Space (Docker SDK).
#   ./scripts/deploy_space.sh <hf-username>/<space-name>
# Requires a one-time login with a write token: `hf auth login --add-to-git-credential`.
#
# The Space gets only what the Docker build needs, taken from the committed code (git archive),
# so no data, artifacts or binary files are uploaded. Hugging Face builds and runs the image;
# the model is trained on synthetic applicants during that build.
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
BUNDLE="$(mktemp -d)"
trap 'rm -rf "$BUNDLE"' EXIT

git archive HEAD Dockerfile .dockerignore pyproject.toml constraints.txt LICENSE src api web | tar -x -C "$BUNDLE"
cp deploy/space-README.md "$BUNDLE/README.md"  # the Space's config header + description

cd "$BUNDLE"
git init --quiet --initial-branch=main
git add .
git commit --quiet -m "Deploy Glassbox demo from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push --force "https://huggingface.co/spaces/$SPACE" main
echo "Pushed. Build log and app: https://huggingface.co/spaces/$SPACE"
