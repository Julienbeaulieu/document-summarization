#!/bin/bash
# set -uo pipefail
set +e

FAILURE=false

echo "safety"
safety check -r requirements.txt -r requirements-dev.txt || FAILURE=true

echo "flake8"
flake8 api src || FAILURE=true

echo "pycodestyle"
pycodestyle api src || FAILURE=true

# echo "pydocstyle"
# pydocstyle api src || FAILURE=true

echo "mypy"
mypy api src || FAILURE=true

echo "bandit"
bandit -ll -r {api,src} || FAILURE=true

# echo "shellcheck"
# shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0