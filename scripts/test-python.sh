#!/usr/bin/env bash
# Run from any directory; keep a separate uv environment for each Python.
set -uo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$root" || exit 1

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [3.10 3.11 ...] [-- PYTEST_ARGS...]"
  echo "Default: 3.10 3.11 3.12 3.13 (3.10 is deprecated)"
  echo "Default suite: core, API, NumPy/JAX, TOML and visualization; no native extension required."
  echo "Use '-- tests' for the full suite, which requires a matching native wheel."
  echo "Each environment is .venv-py<VERSION>; existing extra packages are preserved."
  echo "Uses uv.lock and test/viz extras; uv may download Python and dependencies."
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required and must be on PATH." >&2
  exit 1
fi

versions=()
pytest_args=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    pytest_args=("$@")
    break
  fi
  version="$1"
  if [[ ! "$version" =~ ^3\.([0-9]+)(\.[0-9]+)?$ ]]; then
    echo "Invalid Python version: $version (expected e.g. 3.12 or 3.12.8)" >&2
    exit 2
  fi
  minor="${BASH_REMATCH[1]}"
  if (( 10#$minor < 10 )); then
    echo "Unsupported Python version: $version (requires Python >=3.10)" >&2
    exit 2
  fi
  versions+=("$version")
  shift
done
if [[ ${#versions[@]} -eq 0 ]]; then
  versions=(3.10 3.11 3.12 3.13)
fi

if [[ ${#pytest_args[@]} -eq 0 ]]; then
  echo "Suite: portable core/API/NumPy/JAX/TOML/viz tests (Rust integration excluded)."
  pytest_args=(
    tests/core tests/api
    tests/test_perturbation_toml.py tests/test_numpy_without_rust.py
    tests/outward/test_dynamics_jax.py
    tests/outward/test_outward_jacobians.py
    tests/outward/test_numerical_diff.py
    tests/outward/test_outward.py
    tests/outward/test_outward_state.py
    tests/outward/test_world_cm_outputs.py
    tests/outward/test_outward_total_gradient.py
    tests/outward/test_cm_dynamics.py
  )
else
  echo "Suite: custom pytest arguments (default suite replaced)."
fi

trap 'exit 130' INT
trap 'exit 143' TERM
results=()
failed=0
for version in "${versions[@]}"; do
  echo
  echo "=== Python $version ==="
  if [[ "$version" == "3.10" || "$version" == 3.10.* ]]; then
    echo "Note: Python 3.10 is a deprecated compatibility target; Python >=3.11 is recommended."
  fi
  UV_PROJECT_ENVIRONMENT="$root/.venv-py$version" \
    uv run --python "$version" --locked --extra test --extra viz \
      --no-default-groups --inexact python -m pytest "${pytest_args[@]}"
  status=$?
  if [[ $status -eq 130 || $status -eq 143 ]]; then
    exit "$status"
  elif [[ $status -eq 0 ]]; then
    results+=("Python $version: PASS")
  else
    results+=("Python $version: FAIL (exit $status; see output above)")
    failed=1
  fi
done

echo
echo "=== Results ==="
printf '%s\n' "${results[@]}"
exit "$failed"
