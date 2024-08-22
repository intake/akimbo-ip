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
    """matches fixed-list[4, u8] and fixed-bytestring[4] and ANY 4-byte value (like uint32, assumed big-endian"""
    return (arr.is_leaf and arr.dtype.itemsize == 4) or (
        arr.is_regular and arr.size == 4 and arr.content.is_leaf and arr.content.dtype.itemsize == 1)


def match_ip6(arr):
    """matches fixed-list[16, u8] and fixed-bytestring[16]"""
    return arr.is_regular and arr.size == 16 and arr.content.is_leaf and arr.content.dtype.itemsize == 1


def match_prefix(arr):
    """A network prefix is always one byte"""
    return arr.is_leaf and arr.dtype.itemsize == 1


def match_net4(arr, address="address", prefix="prefix"):
    """Matches a record with IP4 field and prefix field (u8)"""
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip4(arr[address])
        and match_prefix(arr[prefix])
    )


def match_net6(arr, address="address", prefix="prefix"):
    return (
        arr.is_record
        and {address, prefix}.issubset(arr.fields)
        and match_ip6(arr[address])
        and match_prefix(arr[prefix])
    )


def match_stringlike(arr):
    return "string" in arr.parameters.get("__array__", "")


def parse_address4(str_arr):
    """Interpret (byte)strings as IPv4 addresses
    
    Output will be fixed length 4 bytestring array
    """
    out = lib.parse4(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return utils.u8_to_ip4(out.view("uint8"))


def parse_address6(str_arr):
    """Interpret (byte)strings as IPv6 addresses
    
    Output will be fixed length 4 bytestring array
    """
    out = lib.parse6(str_arr.offsets.data.astype("uint32"), str_arr.content.data)
    return utils.u8_to_ip6(out.view("uint8"))


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


def to_ip4(arr):
    if arr.is_leaf:
        return arr.data.view("uint32"),
    else:
        # bytestring or 4 * uint8 regular
        return arr.content.data.view("uint32"),


def to_ip6(arr):
    # always pass as bytes, and assume length is mod 16 in rust
    return arr.content.data.view("uint8"),
    

def dec_ip(func, conv=to_ip4, match=match_ip4, outtype=ak.contents.NumpyArray):
    @functools.wraps(func)
    def func1(arr):
        return func(*conv(arr))

    return dec(func1, match=match, outtype=outtype, inmode="awkward")


class IPAccessor:
    def __init__(self, accessor) -> None:
        self.accessor = accessor

    is_unspecified4 = dec_ip(lib.is_unspecified4)
    is_broadcast4 = dec_ip(lib.is_broadcast4)
    is_global4 = dec_ip(lib.is_global4)
    is_loopback4 = dec_ip(lib.is_loopback4)
    is_private4 = dec_ip(lib.is_private4)
    is_link_local4 = dec_ip(lib.is_link_local4)
    is_shared4 = dec_ip(lib.is_shared4)
    is_benchmarking4 = dec_ip(lib.is_benchmarking4)
    is_reserved4 = dec_ip(lib.is_reserved4)
    is_multicast4 = dec_ip(lib.is_multicast4)
    is_documentation4 = dec_ip(lib.is_documentation4)

    to_string4 = dec_ip(lib.to_text4, outtype=utils.to_ak_string)

    parse_address4 = dec(parse_address4, inmode="ak", match=match_stringlike)
    parse_net4 = dec(parse_net4, inmode="ak", match=match_stringlike)
    
    contains4 = dec(contains4, inmode="ak", match=match_net4)

    to_ipv6_mapped = dec_ip(lib.to_ipv6_mapped, outtype=utils.u8_to_ip6)

    hosts4 = dec(hosts4, match=match_net4, inmode="ak")

    is_benchmarking6 = dec_ip(lib.is_benchmarking6, conv=to_ip6, match=match_ip6)
    is_global6 = dec_ip(lib.is_global6, conv=to_ip6, match=match_ip6)
    is_documentation6 = dec_ip(lib.is_documentation6, conv=to_ip6, match=match_ip6)
    is_unspecified6 = dec_ip(lib.is_unspecified6, conv=to_ip6, match=match_ip6)
    is_loopback6 = dec_ip(lib.is_loopback6, conv=to_ip6, match=match_ip6)
    is_multicast6 = dec_ip(lib.is_multicast6, conv=to_ip6, match=match_ip6)
    is_unicast6 = dec_ip(lib.is_unicast6, conv=to_ip6, match=match_ip6)
    is_ipv4_mapped = dec_ip(lib.is_ipv4_mapped, conv=to_ip6, match=match_ip6)
    is_unicast_link_local = dec_ip(lib.is_unicast_link_local, conv=to_ip6, match=match_ip6)
    is_unique_local = dec_ip(lib.is_unique_local, conv=to_ip6, match=match_ip6)

    to_string6 = dec_ip(lib.to_text6, conv=to_ip6, match=match_ip6, outtype=utils.to_ak_string)
    parse_address6 = dec(parse_address6, inmode="ak", match=match_stringlike)


Accessor.register_accessor("ip", IPAccessor)
