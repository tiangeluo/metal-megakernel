"""Synchronous Metal buffer handoff between MLX and the two native attention schedules.

MLX exports a DLPack kDLMetal data field as an id<MTLBuffer> plus byte_offset.
The producing MLX stream is synchronized first. NativeRunner waits for its command
buffer before MLX consumes its output. Only a private, exclusively allocated output
tensor is externally written; token projections are read-only.
"""

import ctypes as ct
import numpy as np


class Device(ct.Structure):
    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


class Dtype(ct.Structure):
    _fields_ = [("code", ct.c_uint8), ("bits", ct.c_uint8), ("lanes", ct.c_uint16)]


class Tensor(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("device", Device),
        ("ndim", ct.c_int),
        ("dtype", Dtype),
        ("shape", ct.POINTER(ct.c_int64)),
        ("strides", ct.POINTER(ct.c_int64)),
        ("byte_offset", ct.c_uint64),
    ]


def metal_view(array, shape):
    """Inspect a borrowed legacy DLManagedTensor. Keep the capsule alive through use."""
    if tuple(array.__dlpack_device__()) != (8, 0):
        raise RuntimeError(
            "A direct Metal DLPack export from MLX device zero is required"
        )
    capsule = array.__dlpack__()
    get = ct.pythonapi.PyCapsule_GetPointer
    get.argtypes = [ct.py_object, ct.c_char_p]
    get.restype = ct.c_void_p
    # Unversioned DLPack exports start with the DLTensor. The capsule owns its lifetime.
    ptr = get(capsule, b"dltensor")
    tensor = ct.cast(ptr, ct.POINTER(Tensor)).contents
    if tensor.device.type != 8 or tensor.device.id != 0 or not tensor.data:
        raise ValueError("The capsule must carry a Metal buffer")
    actual = tuple(tensor.shape[: tensor.ndim])
    if actual != tuple(shape) or (
        tensor.dtype.code,
        tensor.dtype.bits,
        tensor.dtype.lanes,
    ) != (2, 16, 1):
        raise ValueError(f"Incorrect shape or precision for native binding: {actual}")
    stride = 1
    for i in range(tensor.ndim - 1, -1, -1):
        if tensor.strides and actual[i] > 1 and tensor.strides[i] != stride:
            raise ValueError("Metal binding requires a row-contiguous tensor")
        stride *= actual[i]
    return capsule, tensor.data, tensor.byte_offset


class MetalAttention:
    """Device-resident persistent KV state. Per-layer host fences are charged to wall time."""

    def __init__(self, mx, method, config, layer, state, position, window, dynamic):
        from native.runner import NativeRunner
        from reference.fixtures import make_token

        b = state["cache_k"].shape[0]
        shapes = make_token(
            np.random.default_rng(931),
            b,
            config.query_heads,
            config.kv_heads,
            config.head_dim,
            config.rel_dim,
        )
        self.runner = NativeRunner(
            shapes,
            layer["attention"],
            state,
            position=position,
            window=window,
            dynamic=dynamic,
        )
        self.method, self.mx = method, mx
        # A separate C-contiguous allocation is owned solely by this adapter. The
        # case synchronizes all consumers before the next run reuses the scratch.
        self.scratch = mx.array(np.zeros(shapes["gate"].shape, dtype=np.float16))
        mx.eval(self.scratch)
        self.scratch_export, ptr, offset = metal_view(self.scratch, self.scratch.shape)
        self.runner.bind_external("out", ptr, offset)

    def __call__(self, token, unused_weights, unused_state):
        # mx.contiguous may preserve zero strides on broadcast constants in some
        # versions. Create local dummy dynamic taps only for the static layer.
        arrays = {
            key: self.mx.contiguous(value)
            for key, value in token.items()
            if self.runner.config.dynamic or key not in ("dk", "dv")
        }
        self.mx.eval(*arrays.values())
        self.mx.synchronize()
        owners = []
        try:
            for key, array in arrays.items():
                cap, ptr, offset = metal_view(array, self.runner.shapes[key])
                owners.append(cap)
                self.runner.bind_external(key, ptr, offset)
            self.runner.run(self.method)
        finally:
            for key in arrays:
                self.runner.bind_external(key, None)
        # Attention state is native-owned. Reading it here would distort timing.
        return self.scratch, {}

    def read_state(self):
        return self.runner.outputs()[1]

    def close(self):
        self.runner.close()
