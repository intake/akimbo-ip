import awkward as ak
import numpy as np
import pyarrow as pa

from akimbo.mixin import Accessor
from akimbo.apply_tree import dec
import akimbo_ip.akimbo_ip as lib
from akimbo_ip import utils


def match_ip4(arr):
    # non-regular is not passed, might not all have right size
    return (arr.is_leaf and arr.dtype.itemsize == 4) or (
        arr.is_regular and arr.size == 4 and arr.content.is_leaf and arr.content.dtype.itemsize == 1)


def match_ip6(arr):
    return arr.is_leaf and arr.dtype.itemsize == 16


def match_net4(arr, address="address", prefix="prefix"):
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip4(arr[address])
    )


def match_stringlike(arr):
    return "string" in arr.parameters.get("__array__", "")


def parse4(str_arr):
    out = lib.parse4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return ak.contents.RegularArray(ak.contents.NumpyArray(out.view("uint8")), size=4)


def dec4(func, match=match_ip4, outtype=ak.contents.NumpyArray):
    def func1(arr):
        if arr.is_regular:
            # already checked list-of-fours
            arr = arr.content.data.view("uint32")
        else:
            arr = arr.data.astype("uint32")
        return func(arr)

    return dec(func1, match=match, outtype=outtype, inmode="awkward")


class IPAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    is_global4 = dec4(lib.is_global4)

    to_string4 = dec4(lib.to_text4, outtype=utils.to_ak_string)

    parse4 = dec(parse4, inmode="ak", match=match_stringlike)

    def parse_net4(self):
        t = self.accessor._obj.values._pa_array.type
        assert t == "string" or t == "binary"
        data = self.accessor._obj.values._pa_array.combine_chunks()
        mask, offsets, by = data.buffers()
        assert mask is None
        out = lib.parsenet4(
            np.frombuffer(offsets, "uint32"), np.frombuffer(by, "uint8")
        )
        return self.accessor.to_output(
            ak.Array(
                {
                    "address": ak.contents.RegularArray(
                        ak.contents.NumpyArray(
                            out[0].view("uint8"), parameters={"__array__": "byte"}
                        ),
                        size=4,
                        parameters={"__array__": "bytestring"},
                    ),
                    "prefix": ak.contents.NumpyArray(out[1]),
                }
            )
        )


Accessor.register_accessor("ip", IPAccessor)
