# Native JAX vs TorchTPU: one imported Pallas attention kernel

This repository compares the **same** JAX/Pallas Flash Attention forward and
backward implementation through three frontends on a single TPU v5e chip:
native JAX, direct PyTorch custom ops, and
`torch.compile(backend="tpu", dynamic=False, fullgraph=True)`. The PyTorch
operator is registered with the public `torch.tpu.pallas.jax_op` API and uses
`register_autograd` for backward. The upstream kernel is installed as a pinned
dependency, not copied into this repository.

The comparison is correctness-gated: every timed case checks output and
Q/K/V gradients against native JAX. Timings are synchronized **host-observed**
latencies, not device-only kernel times. This is a bounded experiment, not a
claim about full-model training, convergence, production stability, or all TPU
shapes.

## Reproduce

Use a TPU v5e VM with Python 3.12 and keep these repositories as siblings.
The upstream commit contains its Apache-2.0 license.

```bash
git clone https://github.com/ashioyajotham/recompute_dont_restore.git
git -C recompute_dont_restore checkout c5bb931da44489f0697d97ff94bddaaffeee94d6
git clone https://github.com/ashioyajotham/torchtpu-pallas-vs-jax.git
cd torchtpu-pallas-vs-jax
bash scripts/bootstrap_tpu_vm.sh
source .venv-tpu/bin/activate
python -m pip check
python -m pytest -q
bash scripts/run_v5e8.sh
RUN_MATRIX=1 bash scripts/run_v5e8.sh
```

The bootstrap pins TorchTPU 0.3.0, PyTorch 2.15.0+cpu, libtpu 0.0.51, and
JAX/JAXlib 0.9.2. The first run performs doctor and smoke gates. The second
runs three fresh-process repetitions for each sequence length (256, 512,
1024, 2048, 4096) and mask (causal/noncausal), plus an adversarial fixture.
The full run requires a clean worktree. The runner defaults to normal Python
interpreter teardown; `TORCH_TPU_SAFE_PROCESS_EXIT=1` is only an opt-in
diagnostic workaround and cannot validate normal teardown.

Raw results are written under ignored `artifacts/`. See
[`docs/methodology.md`](docs/methodology.md) for measurement and interpretation
details.

## Verified public-wheel result (November 2026)

The [TorchTPU 0.3.0 v5e result](results/public-torchtpu-0.3.0-v5e.json)
passed ten separate doctor runs with normal Python interpreter exit, a smoke
case, an adversarial case, and the complete 10-shape/mask matrix with three
fresh-process repetitions (30 case runs). All recorded output and Q/K/V
gradient comparisons passed; the largest recorded absolute difference was
0.0 for this fixed fixture and dtype. This does not establish general
numerical equivalence.

For noncausal attention, the median of three paired TorchTPU/native-JAX
**host-latency ratios** was:

| Sequence | Direct forward | Compiled forward | Direct forward+backward | Compiled forward+backward |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.64× | 1.51× | 2.88× | 2.20× |
| 1024 | 1.36× | 1.28× | 1.39× | 1.24× |
| 4096 | 1.04× | 1.03× | 1.06× | 1.05× |

The JSON contains the causal rows, S=512/2048, absolute timings, and
repetition ranges. These ratios reflect the full synchronized Python call;
they do not isolate the TPU kernel or explain the source of the remaining
overhead. Forward-plus-backward is a separately timed path, not an isolated
backward measurement. The result is tied to comparison source commit
`0dae440e584be34a0ca72dc405dc3a69a6fc652b`, upstream commit
`c5bb931da44489f0697d97ff94bddaaffeee94d6`, and the exact public
TorchTPU wheel SHA-256 recorded in the JSON. The published result contains
allowlisted aggregate fields only; raw machine logs remain outside this
repository.

## Scope and attribution

The original comparison harness is Apache-2.0 licensed. The upstream
`recompute_dont_restore` package is a separate project with its own license.
The [TorchTPU Sprint sample](https://github.com/ashioyajotham/torchtpu-pallas-attention-sprint)
is a separate model-level integration project; its direct-versus-compiled
latencies must not be mixed with this native-JAX comparison.
