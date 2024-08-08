import awkward as ak
import numpy as np
import pyarrow as pa


styp = pa.string()


def pa_mask(mask):
    """Make pyarrow bitpacked buffer for use as a mask"""
    if mask is not None:
        if mask.dtype == "bool":
            mask = np.packbits(mask, bitorder="little")
        mask = pa.py_buffer(mask)
    return mask


def to_pa_string(data, offsets, mask=None):
    """Construct pyarrow string-type array from components

    data: u8 array
    offsets: u32 or u64 array
        Length is number of elements + 1; first value is 0, last value is
        length of data, and values always increase
    mask: bool bitpacked array
        Length is the number of elements / 8, rounded up. 1 is valid.
    """
    mask = pa_mask(mask)
    return pa.Array.from_buffers(
        styp, len(offsets) - 1, [mask, pa.py_buffer(offsets), pa.py_buffer(data)]
    )


form = ak.forms.ListOffsetForm(
    "i32",
    ak.forms.NumpyForm("uint8", parameters={"__array__": "char"}),
    parameters={"__array__": "string"},
)


def to_ak_string(inputs, highlevel=False):
    """Make awkward string array from bytes/offsets arrays"""
    data, offsets = inputs
    return ak.from_buffers(
        form,
        len(offsets) - 1,
        {"None-offsets": offsets, "None-data": data},
        highlevel=highlevel,
    )


def u8_to_ip6(arr):
    """Make fixed-length bytestrings for IPv6 output"""
    return ak.contents.RegularArray(
        ak.contents.NumpyArray(arr.view('uint8'), parameters={"__array__": "byte"}),
        size=16,
        parameters={"__array__": "bytestring"}
    )


def u8_to_ip4(arr):
    """Make fixed-length bytestrings for IPv6 output"""
    return ak.contents.RegularArray(
        ak.contents.NumpyArray(arr.view("uint8"), parameters={"__array__": "byte"}),
        size=4,
        parameters={"__array__": "bytestring"}
    )
    