#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

run_manifest() {
  local manifest="$1"
  echo "[paper] running manifest: ${manifest}"
  "${PYTHON_BIN}" scripts/run_matrix.py --manifest "${manifest}"
}

aggregate_results() {
  local manifest="$1"
  local section
  section="$(basename "${manifest}" .yaml)"
  echo "[paper] aggregating results for ${section}"
  "${PYTHON_BIN}" scripts/aggregate_results.py --section "${section}"
}

export_artifacts() {
  echo "[paper] exporting paper artifacts"
  "${PYTHON_BIN}" scripts/export_paper_artifacts.py
}

run_section() {
  local manifest="$1"
  run_manifest "${manifest}"
  aggregate_results "${manifest}"
  export_artifacts
}

note_outputs() {
  echo "[paper] aggregates: outputs/aggregates"
  echo "[paper] tables:     outputs/tables"
  echo "[paper] figures:    outputs/figures"
}
