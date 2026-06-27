"""Triton elementwise fusion kernels for decode projection side work."""

from .copy_add_mul import triton_add_mul, triton_copy_add_mul

__all__ = ["triton_add_mul", "triton_copy_add_mul"]
