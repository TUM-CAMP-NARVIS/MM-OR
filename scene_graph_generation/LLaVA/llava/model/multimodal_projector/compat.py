"""
Backend selection for the ops MM-OR used to get from compiled third-party packages.

`torch_scatter` is used if present (so the frozen baseline env keeps bit-identical
behaviour), otherwise the native ATen implementation in native_ops.py takes over. The
native path is validated against torch_scatter by the mssg parity harness (parity/ptv3_parity.py): exact for
reduce="max", ~1e-7 for reduce="mean".

This indirection is what lets mmor-modern and the Holoscan env drop torch_scatter
entirely while the baseline stays untouched as the parity oracle.
"""

try:
    from torch_scatter import segment_csr  # noqa: F401
    SEGMENT_CSR_BACKEND = "torch_scatter"
except ImportError:  # pragma: no cover - depends on the environment
    from .native_ops import segment_csr  # noqa: F401
    SEGMENT_CSR_BACKEND = "native"
