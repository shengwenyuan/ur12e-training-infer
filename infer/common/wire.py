"""Bounded MessagePack transport; no executable deserialization."""

# Protocol numbers must exclude bool, which is an int subclass.
# pylint: disable=unidiomatic-typecheck


import msgpack
import numpy as np

MAX_BYTES = 16 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object arrays are prohibited")
        return {
            "__array__": True,
            "dtype": value.dtype.str,
            "shape": value.shape,
            "data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if value.get("__array__") is True:
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if (
            dtype.hasobject
            or dtype.kind not in "biuf"
            or len(shape) > 4
            or any(type(n) is not int or n < 0 or n > MAX_BYTES for n in shape)
        ):
            raise ValueError("invalid wire array")
        count = 1
        for size in shape:
            count *= size
        if (
            count * dtype.itemsize != len(value["data"])
            or len(value["data"]) > MAX_BYTES
        ):
            raise ValueError("invalid wire array size")
        return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()
    return value


def pack(value):
    """Serialize one request or response within the transport size budget."""
    data = msgpack.packb(value, default=_encode, use_bin_type=True)
    if len(data) > MAX_BYTES:
        raise ValueError("message too large")
    return data


def unpack(data):
    """Reject oversize payloads before decoding."""
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError("invalid message")
    return msgpack.unpackb(
        data,
        object_hook=_decode,
        raw=False,
        max_array_len=MAX_BYTES,
        max_map_len=10000,
    )
