import ipaddress
import functools

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
    return arr.is_regular and arr.size == 16 and arr.content.is_leaf and arr.content.dtype.itemsize == 1


def match_net4(arr, address="address", prefix="prefix"):
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip4(arr[address])
    )


def match_net6(arr, address="address", prefix="prefix"):
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip6(arr[address])
    )


def match_stringlike(arr):
    return "string" in arr.parameters.get("__array__", "")


def parse_address4(str_arr):
    """Interpret (byte)strings as IPv4 addresses
    
    Output will be fixed length 4 bytestring array
    """
    out = lib.parse4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return utils.u8_to_ip4(out.view("uint8"))


def parse_net4(str_arr):
    """Interpret (byte)strings as IPv4 networks (address/prefix)
    
    Output will be a record array {"address": fixed length 4 bytestring, "prefix": uint8}
    """
    out = lib.parsenet4(
        str_arr.offsets.data.astype("uint32"), str_arr.content.data
    )
    return ak.contents.RecordArray(
        [ak.contents.RegularArray(
            ak.contents.NumpyArray(out[0].view("uint8"), parameters={"__array__": "byte"}), 
            size=4, 
            parameters={"__array__": "bytestring"}
        ),
        ak.contents.NumpyArray(out[1])],
        fields=["address", "prefix"]
    )
    

def contains4(nets, other, address="address", prefix="prefix"):
    # TODO: this is single-value only
    arr = nets[address]
    if arr.is_leaf:
        arr = arr.data.astype("uint32")
    else:
        # fixed bytestring or 4 * uint8 regular
        arr = arr.content.data.view("uint32")
    ip = ipaddress.IPv4Address(other)._ip
    out = lib.contains_one4(arr, nets[prefix].data.astype("uint8"), ip)
    return ak.contents.NumpyArray(out)


def hosts4(nets, address="address", prefix="prefix"):
    arr = nets[address]
    if arr.is_leaf:
        arr = arr.data.astype("uint32")
    else:
        # fixed bytestring or 4 * uint8 regular
        arr = arr.content.data.view("uint32")
    ips, offsets = lib.hosts4(arr, nets[prefix].data.astype("uint8"))
    return ak.contents.ListOffsetArray(
        ak.index.Index64(offsets),
        utils.u8_to_ip4(ips)
    )


def dec4(func, match=match_ip4, outtype=ak.contents.NumpyArray):
    @functools.wraps(func)
    def func1(arr):
        if arr.is_leaf:
            arr = arr.data.astype("uint32")
        else:
            # bytestring or 4 * uint8 regular
            arr = arr.content.data.view("uint32")
        return func(arr)

    return dec(func1, match=match, outtype=outtype, inmode="awkward")


class IPAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    is_unspecified4 = dec4(lib.is_unspecified4)
    is_broadcast4 = dec4(lib.is_broadcast4)
    is_global4 = dec4(lib.is_global4)
    is_loopback4 = dec4(lib.is_loopback4)
    is_private4 = dec4(lib.is_private4)
    is_link_local4 = dec4(lib.is_link_local4)
    is_shared4 = dec4(lib.is_shared4)
    is_benchmarking4 = dec4(lib.is_benchmarking4)
    is_reserved4 = dec4(lib.is_reserved4)
    is_multicast4 = dec4(lib.is_multicast4)
    is_documentation4 = dec4(lib.is_documentation4)

    to_string4 = dec4(lib.to_text4, outtype=utils.to_ak_string)

    parse_address4 = dec(parse_address4, inmode="ak", match=match_stringlike)

    parse_net4 = dec(parse_net4, inmode="ak", match=match_stringlike)
    
    contains4 = dec(contains4, inmode="ak", match=match_net4)

    to_ipv6_mapped = dec(lib.to_ipv6_mapped, inmode="numpy", match=match_ip4, 
                         outtype=utils.u8_to_ip6)

    hosts4 = dec(hosts4, match=match_net4, inmode="ak")

Accessor.register_accessor("ip", IPAccessor)
