#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_TPU_VERSION=${TORCH_TPU_VERSION:-0.3.0}
TORCH_VERSION=${TORCH_VERSION:-2.15.0+cpu}
LIBTPU_VERSION=${LIBTPU_VERSION:-0.0.51}
JAX_VERSION=${JAX_VERSION:-0.9.2}
UPSTREAM_REPO=${UPSTREAM_REPO:-"$(dirname "${REPO_ROOT}")/recompute_dont_restore"}
UPSTREAM_SHA=${UPSTREAM_SHA:-c5bb931da44489f0697d97ff94bddaaffeee94d6}
VENV_DIR=${VENV_DIR:-"${REPO_ROOT}/.venv-tpu"}
WHEEL_DIR=${WHEEL_DIR:-"${REPO_ROOT}/artifacts/wheels"}

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
mkdir -p "${WHEEL_DIR}"

if [[ -n ${TORCH_TPU_WHEEL:-} ]]; then
  if [[ ! -f ${TORCH_TPU_WHEEL} || ${TORCH_TPU_WHEEL} != *.whl ]]; then
    echo "TORCH_TPU_WHEEL must be an existing full .whl path" >&2
    exit 2
  fi
  cp "$(realpath "${TORCH_TPU_WHEEL}")" "${WHEEL_DIR}/"
  WHEEL_PATH="${WHEEL_DIR}/$(basename "${TORCH_TPU_WHEEL}")"
else
  python -m pip download --no-deps \
    --dest "${WHEEL_DIR}" \
    "torch_tpu==${TORCH_TPU_VERSION}"
  WHEEL_PATH=$(find "${WHEEL_DIR}" -maxdepth 1 -type f \
    -name "torch_tpu-${TORCH_TPU_VERSION}-*.whl" -print -quit)
fi
if [[ -z ${WHEEL_PATH:-} || ! -f ${WHEEL_PATH} ]]; then
  echo "Unable to resolve the pinned TorchTPU wheel" >&2
  exit 2
fi

# Keep versions pinned so this comparison run has unambiguous provenance.
python -m pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "libtpu==${LIBTPU_VERSION}"
python -m pip install "${WHEEL_PATH}"
python -m pip install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
python -m pip install -e '.[test]'

if [[ ! -f ${UPSTREAM_REPO}/pyproject.toml ]]; then
  echo "UPSTREAM_REPO is not a cloned recompute_dont_restore project: ${UPSTREAM_REPO}" >&2
  exit 2
fi
UPSTREAM_HEAD=$(git -C "${UPSTREAM_REPO}" rev-parse HEAD)
if [[ ${UPSTREAM_HEAD} != "${UPSTREAM_SHA}"* ]]; then
  echo "Upstream HEAD ${UPSTREAM_HEAD} does not match required revision ${UPSTREAM_SHA}" >&2
  exit 2
fi
python -m pip install --no-deps -e "${UPSTREAM_REPO}"
python -m pip check

echo "Bootstrap complete."
echo "TorchTPU wheel: ${WHEEL_PATH}"
echo "TorchTPU wheel SHA-256: $(sha256sum "${WHEEL_PATH}" | cut -d ' ' -f1)"
echo "Upstream HEAD: ${UPSTREAM_HEAD}"
echo "Next: bash scripts/run_v5e8.sh"
