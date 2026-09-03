# Release test report

Release: `pi-MNovo-v0.1.0`
Test date: 2026-09-03
Platform: Linux x86-64, Python 3.10.21, PyTorch 2.5.1+cu124

## Clean installation

A new environment was created from `environment.yml` with no packages inherited
from the development environment. `pip check` reported no broken requirements.
The bundled `ctcdecode` extension imported successfully and its ELF RPATH was
verified as relative to the active environment's PyTorch libraries.

## Automated checks

- Ruff static check: passed.
- Pytest: 16 passed.
- Unified checkpoint SHA256: passed.
- Embedded component hashes and required-component inventory: passed.
- Backbone repackaging audit: all 286 state tensors were bitwise equal.
- Private path and credential-pattern scan: no findings.

## End-to-end smoke test

The installed `pi-mnovo` entry point processed the public two-spectrum fixture
`tests/data/synthetic.mgf` with the backbone forced to CPU on a CUDA-capable
host. It materialized the MGF input, loaded the
unified checkpoint, generated Beam5 plus PMC candidates, ran sequence routing,
and wrote two predictions. The output contained the documented nine-column TSV
schema and four-decimal confidence scores. This diagnostic does not establish
CPU-only support because precise-mass-control decoding uses CUDA through CuPy;
normal inference should use CUDA throughout.

Warnings emitted by PyTorch Lightning concern its deprecated `pkg_resources`
namespace API. `setuptools` is pinned below version 81 to retain that API for the
frozen PyTorch Lightning 1.8.6 runtime.
