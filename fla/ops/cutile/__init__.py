try:
    import cupy as cp
    import cuda.tile as ct
    HAS_CUTILE = True
except ImportError:
    HAS_CUTILE = False

__all__ = [
    "HAS_CUTILE",
]
