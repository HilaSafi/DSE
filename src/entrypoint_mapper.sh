#!/usr/bin/env bash
set -e

# Activate conda env implicitly via PATH (already set in Dockerfile)
cd /opt/DSE

echo ""
echo "Mapper container ready (conda env: mapper_experiments, Python 3.10)"
echo "Repo: /opt/DSE"
echo ""

# Non-interactive default
if [[ "${RUN_DEFAULT:-}" == "yes" ]]; then
  echo "RUN_DEFAULT=yes -> running default mapper experiment..."
  python -u src/run_mapper_experiments.py
  exit 0
fi

read -r -p "Run default mapper experiments now? [y/N]: " ans
if [[ "$ans" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  python -u src/run_mapper_experiments.py
else
  echo "Dropping you into a shell. You're in /opt/DSE."
  exec "${@:-bash}"
fi
