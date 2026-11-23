# Comparison methodology

The fixture fixes batch=1, heads=1, head dimension=128, BF16 operands, and
static sequence lengths 256–4096. Each native/direct/compiled worker runs in a
fresh process. Every case uses five warmups and 30 synchronized host-timed
trials; the published value is the median of three fresh-process p50 values.
Forward and forward-plus-backward are separately timed. The latter is **not**
an isolated backward measurement, and subtracting their p50s is not valid.

The correctness gate compares attention output and dQ/dK/dV on the same
fixture. A case only contributes timings when the comparison passes. The
aggregate rejects missing cases and incorrect repetition counts. Both causal
and noncausal cases run; causal executed FLOPs and useful FLOPs are kept
distinct. Reported ratios are TorchTPU/native JAX host-latency ratios, not
device-only kernel speedups or model FLOPs utilization.

The script records versions, wheel SHA-256, source commits, flags, and raw
logs in ignored `artifacts/`. Those logs can contain local paths and should
not be published verbatim. A public result should include the exact tested
stack, code revision, aggregate and correctness status, and any abnormal
process exits. A successful report written before a teardown crash does not
count as a normal-exit pass.

`scripts/export_public_results.py` builds an allowlisted JSON summary from a
completed private run. It requires ten clean doctor exits, the passing smoke,
adversarial, and 30-case matrix gates, consistent source and wheel provenance,
and `TORCH_TPU_SAFE_PROCESS_EXIT=0` in every runner manifest. It never copies
raw logs, environment paths, package freezes, or storage URIs into the public
result. The public summary is derived evidence; the source reports remain
private and can be provided to maintainers on request.

The November 2026 result was produced with the public TorchTPU 0.3.0 wheel,
PyTorch 2.15.0+cpu, libtpu 0.0.51, and JAX/JAXlib 0.9.2. The ten-run doctor
check used separate Python processes with normal interpreter teardown; all
ten returned zero and wrote passing reports. The full sweep then exited zero
after all 30 shape/mask/repetition case runs and the adversarial gate. The
ratio for each repetition is that repetition's TorchTPU p50 divided by its
native-JAX p50; the table reports the median of those three paired ratios.
This measurement does not establish a cause for any latency difference, and
the fix for the earlier teardown crash is not being attributed to a PJRT API
version change.

The TPU run used comparison source commit
`0dae440e584be34a0ca72dc405dc3a69a6fc652b` in the private source
repository. This public repository was created from an allowlisted export
without carrying over that repository's history. Its executable comparison
and binding modules match the tested source; the exported doctor default
selects the tested 0.3.0 wheel, and the optional process-exit workaround has
documentation-only wording changes. A test of the published Git revision on
TPU has not been performed.
