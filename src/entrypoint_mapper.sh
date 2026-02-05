#!/usr/bin/env bash
set -euo pipefail
cd /opt/DSE

echo ""
echo "Mapper container ready (conda env: mapper_experiments)"
conda run -n mapper_experiments python -c "import sys; print('Python:', sys.executable)"
conda run -n mapper_experiments python -c "import numpy; print('NumPy:', numpy.__version__)"
echo ""

if [[ "${RUN_DEFAULT:-}" == "yes" ]]; then
  exec conda run --no-capture-output -n mapper_experiments python -u src/run_mapper_experiments.py
fi

read -r -p "Run default mapper experiments now? [y/N]: " ans
if [[ "$ans" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  exec conda run --no-capture-output -n mapper_experiments python -u src/run_mapper_experiments.py
else
  echo "Dropping you into an env shell."
  exec conda run --no-capture-output -n mapper_experiments bash
fi
