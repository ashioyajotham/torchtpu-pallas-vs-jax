#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-"${REPO_ROOT}/.venv-tpu"}
UPSTREAM_REPO=${UPSTREAM_REPO:-"$(dirname "${REPO_ROOT}")/recompute_dont_restore"}
UPSTREAM_SHA=${UPSTREAM_SHA:-c5bb931da44489f0697d97ff94bddaaffeee94d6}
TORCH_TPU_VERSION=${TORCH_TPU_VERSION:-0.3.0}
RUN_MATRIX=${RUN_MATRIX:-0}
REPETITIONS=${REPETITIONS:-3}
TORCH_TPU_SAFE_PROCESS_EXIT=${TORCH_TPU_SAFE_PROCESS_EXIT:-0}
cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
export COMPARISON_REPO="${REPO_ROOT}"
export UPSTREAM_REPO UPSTREAM_SHA TORCH_TPU_VERSION TORCH_TPU_SAFE_PROCESS_EXIT
TORCH_TPU_WHEEL=$(find "${REPO_ROOT}/artifacts/wheels" -maxdepth 1 -type f -name "torch_tpu-${TORCH_TPU_VERSION}-*.whl" -print -quit)
export TORCH_TPU_WHEEL
if [[ -z ${TORCH_TPU_WHEEL} || ! -f ${TORCH_TPU_WHEEL} ]]; then
  echo "Pinned wheel artifact not found; run bash scripts/bootstrap_tpu_vm.sh first" >&2
  exit 2
fi
if [[ -n ${JAX_INTERPRET_PALLAS:-} ]]; then echo "JAX_INTERPRET_PALLAS must be unset" >&2; exit 2; fi
if [[ -n ${TPU_LAUNCH_BLOCKING:-} || -n ${TPU_DEFER_AND_FUSE:-} ]]; then
  echo "Unset TPU_LAUNCH_BLOCKING and TPU_DEFER_AND_FUSE" >&2; exit 2
fi
tpu-compare doctor --expected-torch-tpu "${TORCH_TPU_VERSION}" --out artifacts/doctor.json
tpu-compare smoke --results artifacts/smoke
if [[ ${RUN_MATRIX} != 1 ]]; then
  echo "Doctor and smoke gates passed. Set RUN_MATRIX=1 for the publication sweep."; exit 0
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "Publication sweep requires a clean comparison worktree; run git status --short" >&2
  exit 2
fi
for repetition in $(seq 1 "${REPETITIONS}"); do
  for causal in false true; do
    causal_flag=(); suffix=noncausal
    if [[ ${causal} == true ]]; then causal_flag=(--causal); suffix=causal; fi
    for seq_len in 256 512 1024 2048 4096; do
      fixture="artifacts/fixtures/s${seq_len}-${suffix}.npz"
      result_dir="artifacts/publication/rep${repetition}/s${seq_len}-${suffix}"
      profile_flag=()
      if [[ ${repetition} == 1 && ${seq_len} == 1024 && ${causal} == true ]]; then profile_flag=(--profile); fi
      tpu-compare fixture --out "${fixture}" --seq-len "${seq_len}" "${causal_flag[@]}"
      tpu-compare suite --fixture "${fixture}" --results "${result_dir}" "${profile_flag[@]}"
    done
  done
done
tpu-compare aggregate --root artifacts/publication --out artifacts/publication/summary.json --expected-repetitions "${REPETITIONS}"

tpu-compare fixture --out artifacts/fixtures/adversarial.npz --batch 1 --heads 1 --seq-len 256 --adversarial
tpu-compare suite --fixture artifacts/fixtures/adversarial.npz --results artifacts/publication/adversarial
