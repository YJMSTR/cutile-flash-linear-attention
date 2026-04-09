# -*- coding: utf-8 -*-
# Adapted from fla/utils.py for PaddlePaddle

import os
import functools

import paddle
import triton

# ===== Environment checks =====
FLA_CI_ENV = os.environ.get('FLA_CI_ENV', '0') == '1'
FLA_CACHE_RESULTS = os.environ.get('FLA_CACHE_RESULTS', '1') == '1'
FLA_DISABLE_TENSOR_CACHE = os.environ.get('FLA_DISABLE_TENSOR_CACHE', '0') == '1'


# ===== Device detection =====
def get_available_device():
    return 'cuda'


def get_multiprocessor_count():
    props = paddle.device.cuda.get_device_properties()
    return props['multi_processor_count']


def _get_device_name():
    return paddle.device.cuda.get_device_name()


device_name = _get_device_name()

IS_NVIDIA = 'nvidia' in device_name.lower() or 'geforce' in device_name.lower() or 'tesla' in device_name.lower()
IS_AMD = 'amd' in device_name.lower() or 'instinct' in device_name.lower()
IS_INTEL = 'intel' in device_name.lower()

try:
    capability = paddle.device.cuda.get_device_capability()
except:
    capability = (0, 0)

IS_NVIDIA_HOPPER = IS_NVIDIA and capability[0] >= 9
IS_NVIDIA_BLACKWELL = IS_NVIDIA and capability[0] >= 10
IS_TF32_SUPPORTED = IS_NVIDIA and capability[0] >= 8
IS_GATHER_SUPPORTED = True
IS_TMA_SUPPORTED = False  # TMA not supported in Paddle migration for now

USE_CUDA_GRAPH = os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1'

# lowercase aliases
is_nvidia = IS_NVIDIA
is_amd = IS_AMD
is_intel = IS_INTEL


# ===== Backend enum for shared memory =====
class Backend:
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400


def check_shared_mem(arch=None, tensor_idx=0):
    """Check if device shared memory meets requirements."""
    try:
        props = paddle.device.cuda.get_device_properties()
        max_smem = props.get('shared_memory_per_block_optin',
                            props.get('shared_memory_per_block', 49152))
    except:
        max_smem = 49152

    if arch is None:
        return max_smem >= Backend.DEFAULT
    elif arch == 'ampere':
        return max_smem >= Backend.AMPERE
    elif arch == 'hopper':
        return max_smem >= Backend.HOPPER
    elif arch == 'ada':
        return max_smem >= Backend.ADA
    return max_smem >= Backend.DEFAULT


def get_all_max_shared_mem():
    try:
        props = paddle.device.cuda.get_device_properties()
        return props.get('shared_memory_per_block_optin',
                        props.get('shared_memory_per_block', 49152))
    except:
        return 49152


# ===== Triton version checks =====
def _check_triton_version(min_version):
    try:
        from importlib.metadata import version
        triton_ver = version('triton')
        from packaging.version import Version
        return Version(triton_ver) >= Version(min_version)
    except:
        return False

TRITON_ABOVE_3_4_0 = _check_triton_version('3.4.0')
TRITON_ABOVE_3_5_1 = _check_triton_version('3.5.1')

# ===== autotune cache =====
SUPPORTS_AUTOTUNE_CACHE = hasattr(triton.autotune, '__wrapped__') or True
try:
    # Check if triton.autotune supports cache_results
    import inspect
    sig = inspect.signature(triton.autotune)
    SUPPORTS_AUTOTUNE_CACHE = 'cache_results' in sig.parameters
except:
    SUPPORTS_AUTOTUNE_CACHE = False

autotune_cache_kwargs = {}
if SUPPORTS_AUTOTUNE_CACHE and FLA_CACHE_RESULTS:
    autotune_cache_kwargs = {'cache_results': True}


# ===== AMP adapters =====
def autocast_custom_fwd(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with paddle.amp.auto_cast(enable=False):
            return fn(*args, **kwargs)
    return wrapper


def autocast_custom_bwd(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with paddle.amp.auto_cast(enable=False):
            return fn(*args, **kwargs)
    return wrapper


# ===== tensor_cache =====
def tensor_cache(fn):
    """Single-entry tensor function cache."""
    if FLA_DISABLE_TENSOR_CACHE:
        return fn

    _cache = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _cache:
            _cache.clear()
            _cache[key] = fn(*args, **kwargs)
        return _cache[key]
    return wrapper


# ===== input_guard =====
def input_guard(fn=None, *, no_guard_contiguous=None):
    """Ensure all tensor inputs are contiguous."""
    if fn is None:
        return functools.partial(input_guard, no_guard_contiguous=no_guard_contiguous)

    skip_names = set(no_guard_contiguous) if no_guard_contiguous else set()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        new_args = []
        # Get parameter names for positional args
        import inspect
        try:
            params = list(inspect.signature(fn).parameters.keys())
        except (ValueError, TypeError):
            params = []

        for i, arg in enumerate(args):
            param_name = params[i] if i < len(params) else ''
            if isinstance(arg, paddle.Tensor) and param_name not in skip_names:
                if not arg.is_contiguous():
                    arg = arg.contiguous()
            new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor) and k not in skip_names:
                if not v.is_contiguous():
                    v = v.contiguous()
            new_kwargs[k] = v
        return fn(*new_args, **new_kwargs)
    return wrapper


def contiguous(fn):
    """Alias for input_guard without parameters."""
    return input_guard(fn)


# ===== checkpoint =====
def checkpoint(fn):
    """Wrap function with recompute."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return paddle.distributed.fleet.utils.recompute(fn, *args, **kwargs)
    return wrapper


# ===== Testing helpers =====
def get_abs_err(x, y):
    return (x - y).abs().max().item()


def get_err_ratio(x, y):
    err = ((x - y).flatten().pow(2).mean().sqrt() /
           (y.flatten().pow(2).mean().sqrt() + 1e-12)).item()
    return err


def assert_close(prefix, ref, tri, ratio, abs_tol=None):
    err = get_err_ratio(ref, tri)
    abs_err = get_abs_err(ref, tri) if abs_tol is not None else None
    msg = f"{prefix} err ratio: {err:.6f}"
    if abs_err is not None:
        msg += f", abs err: {abs_err:.6f}"
    if FLA_CI_ENV:
        ratio = ratio * 2
    assert err < ratio, msg
    if abs_tol is not None:
        assert abs_err < abs_tol, f"{prefix} abs err {abs_err} >= {abs_tol}"
