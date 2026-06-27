"""Triton GEMV kernels for decode-like small-batch linear projection."""

from decode_gemv.kernels.gemv_triton.gemv import triton_gemv

__all__ = ["triton_gemv"]
