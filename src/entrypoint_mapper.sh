#!/usr/bin/env bash
set -euo pipefail

cd /opt/DSE

echo ""
echo "Mapper container ready (conda env: mapper_experiments, Python 3.10)"
echo "Repo: /opt/DSE"
echo ""

# Debug (proves which Python is used)
conda run -n mapper_experiments python -c "import sys; print('Python:', sys.executable)"
conda run -n mapper_experiments python -c "import pandas as pd; print('pandas OK', pd.__version__)"

if [[ "${RUN_DEFAULT:-}" == "yes" ]]; then
  echo "RUN_DEFAULT=yes -> running default mapper experiment..."
  exec conda run --no-capture-output -n mapper_experiments python -u src/run_mapper_experiments.py
fi

read -r -p "Run default mapper experiments now? [y/N]: " ans
if [[ "$ans" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  exec conda run --no-capture-output -n mapper_experiments python -u src/run_mapper_experiments.py
else
  echo "Dropping you into a shell in the mapper env."
  exec conda run --no-capture-output -n mapper_experiments bash
fi
